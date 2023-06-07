from typing import NamedTuple, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap
from jax.numpy import ndarray

from .kernels import BaseKernel
from .utils import matmul_diag


class SparseCovar(NamedTuple):
    U_ref: ndarray
    U_inv: ndarray
    diag: ndarray
    proj_labs: ndarray

class FullCovar(NamedTuple):
    k_nn: ndarray
    y_data: ndarray

def full_covariance_matrix(X_split: Tuple[ndarray, ndarray], Y_data: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> FullCovar:
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
    
    diag = jnp.diag_indices(len(K_NN))
    K_NN = K_NN.at[diag].add(noise**2)

    K_NN = jsp.linalg.cholesky(K_NN)
    return FullCovar(K_NN, Y_data)

def sparse_covariance_matrix(X_split: Tuple[ndarray, ndarray], Y_data: ndarray, X_ref: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> SparseCovar:
    '''Calculates the sparse covariance matrix over the input samples in X_split 
    and the projected input labels in Y_data according to FITC.

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
        sparse (FITC) covariance matrix, projected input labels
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

    # (1 / sqrt(fitc_diag))@V.T
    V_scaled = matmul_diag(1 / jnp.sqrt(fitc_diag), V.T).T

    U_inv = jsp.linalg.cholesky((V_scaled@V_scaled.T).at[diag].add(1.0))

    projected_label = jsp.linalg.solve_triangular(U_inv.T, V@(Y_data / fitc_diag), lower=True)

    return SparseCovar(U_ref, U_inv, fitc_diag, projected_label)

def CovVectorID(X: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Calculates the covariance of each point in X with itself

    Parameters
    ----------
    X : ndarray
        array of shape (N, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N, ), [K(x, x) for x in X]
    '''
    func = lambda v: kernel.eval(v, v, params)
    func = vmap(func, in_axes=(0))
    return func(X)

def CovMatrixFF(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on inputs representing values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1, N2), [K(x1, x2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.eval(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

def CovMatrixFD(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 representing values of the target function and X2
    representing gradient values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1, N2 * n_features), [dK(x1, x2) / dx2 for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.grad2(v1, v2, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return vmap(jnp.ravel, in_axes=0)(func(X1, X2))

def CovMatrixDD(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 and X2 representing derivative values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1 * n_features, N2 * n_features), [dK(x1, x2) / (dx1*dx2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.jac(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return jnp.hstack(jnp.hstack((*func(X1, X2),)))