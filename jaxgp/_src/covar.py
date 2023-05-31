from typing import Tuple, Union, NamedTuple

import jax.numpy as jnp
from jax import vmap
from jax.numpy import ndarray
import jax.scipy as jsp

from .kernels import BaseKernel
from .utils import CovMatrixDD, CovMatrixFD, CovMatrixFF, _build_xT_Ainv_x

class SparseCovar(NamedTuple):
    U_ref: ndarray
    U_inv: ndarray
    diag: ndarray
    proj_labs: ndarray

class FullCovar(NamedTuple):
    k_nn: ndarray

def full_covariance_matrix(X_split: Tuple[ndarray, ndarray], noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> FullCovar:
    '''Calculates the full covariance matrix over the input samples in X_split.

    Parameters
    ----------
    X_split : list[ndarray]
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    noise : Union[float, ndarray]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split.
        ndarray is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        Kernel parameters

    Returns
    -------
    ndarray
        full covariance matrix
    '''
    # Build the full covariance Matrix between all datapoints in X_data depending on if they   
    # represent function evaluations or derivative evaluations
    KF = CovMatrixFF(X_split[0], X_split[0], kernel, params)
    KD = CovMatrixFD(X_split[0], X_split[1], kernel, params)
    KDD = CovMatrixDD(X_split[1], X_split[1], kernel, params)

    K_NN = jnp.vstack((jnp.hstack((KF,KD)), 
                       jnp.hstack((KD.T,KDD))))
    
    # additional small diagonal element added for 
    # numerical stability of the inversion and determinant
    diag = jnp.diag_indices(len(K_NN))
    K_NN = K_NN.at[diag].add(noise**2)

    K_NN, _ = jsp.linalg.cho_factor(K_NN)
    # K_NN = jsp.linalg.cholesky(K_NN)
    return FullCovar(K_NN)

def sparse_covariance_matrix(X_split: Tuple[ndarray, ndarray], Y_data: ndarray, X_ref: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> SparseCovar: #-> Tuple[ndarray, ndarray]:
    '''Calculates the sparse covariance matrix over the input samples in X_split 
    and the projected input labels in Y_data according to the Projected Process Approximation.

    Parameters
    ----------
    X_split : list[ndarray]
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : ndarray
        ndarray of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    X_ref : ndarray
        ndarray of shape (n_referencepoints, n_features). Reference points onto which the whole input dataset is projected.
    noise : Union[float, ndarray]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        ndarray is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        Kernel parameters

    Returns
    -------
    Tuple[ndarray, ndarray]
        sparse (PPA) covariance matrix, projected input labels
    '''
    # Hardcoded squared noise between the reference points
    noise_ref = 1e-2

    # calculates the covariance between the training points and the reference points
    KF = CovMatrixFF(X_ref, X_split[0], kernel, params)
    KD = CovMatrixFD(X_ref, X_split[1], kernel, params)
    K_MN = jnp.hstack((KF,KD))

    # calculates the covariance between each pair of reference points
    K_ref = CovMatrixFF(X_ref, X_ref, kernel, params)
    diag = jnp.diag_indices(len(K_ref))
    K_ref = K_ref.at[diag].add(noise_ref**2)

    # upper cholesky factor of K_ref || U_ref.T@U_ref = K_ref
    U_ref = jsp.linalg.cholesky(K_ref)

    # V is solution to U_ref.T@V = K_MN
    V = jsp.linalg.solve_triangular(U_ref.T, K_MN, lower=True)

    # diag(K_NN)
    func = vmap(lambda v: kernel.eval(v, v, params), in_axes=(0))(X_split[0])
    der = jnp.ravel(vmap(lambda v: jnp.diag(kernel.jac(v, v, params)), in_axes=(0))(X_split[1]))
    K_NN_diag = jnp.hstack((func, der))
    # diag(Q_NN)
    Q_NN_diag = vmap(lambda x: x.T@x, in_axes=(1,))(V)    
    # diag(K_NN) + noise**2 - diag(Q_NN)
    fitc_diag = K_NN_diag + noise**2 - Q_NN_diag
    
    def _mul_diag(diag, matrix):
        return diag*matrix
    mul_diag = vmap(_mul_diag, in_axes=(0,0))

    # (1 / sqrt(fitc_diag))@V.T
    V_scaled = mul_diag(1 / jnp.sqrt(fitc_diag), V.T).T

    U_inv = jsp.linalg.cholesky((V_scaled@V_scaled.T).at[diag].add(1.0))

    projected_label = jsp.linalg.solve_triangular(U_inv.T, V@(Y_data / fitc_diag), lower=True)

    return SparseCovar(U_ref, U_inv, fitc_diag, projected_label)