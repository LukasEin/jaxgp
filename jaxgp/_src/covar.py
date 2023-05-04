from typing import Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy import ndarray

from .kernels import BaseKernel
from .utils import _CovMatrix_Grad, _CovMatrix_Hess, _CovMatrix_Kernel


@jit
def full_covariance_matrix(X_split: Tuple[ndarray, ndarray], noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> ndarray:
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
    KF = _CovMatrix_Kernel(X_split[0], X_split[0], kernel, params)
    KD = vmap(jnp.ravel, in_axes=0)(_CovMatrix_Grad(X_split[0], X_split[1], kernel, params))
    KDD = jnp.hstack(jnp.hstack((*_CovMatrix_Hess(X_split[1], X_split[1], kernel, params),)))

    K_NN = jnp.vstack((jnp.hstack((KF,KD)), 
                       jnp.hstack((KD.T,KDD))))

    # additional small diagonal element added for 
    # numerical stability of the inversion and determinant
    return (jnp.eye(len(K_NN)) * (noise**2 + 1e-6) + K_NN)

@jit
def sparse_covariance_matrix(X_split: Tuple[ndarray, ndarray], Y_data: ndarray, X_ref: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
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
    # calculates the covariance between the training points and the reference points
    KF = _CovMatrix_Kernel(X_ref, X_split[0], kernel, params)
    KD = vmap(jnp.ravel, in_axes=0)(_CovMatrix_Grad(X_ref, X_split[1], kernel, params))
    
    K_MN = jnp.hstack((KF,KD))

    # calculates the covariance between each pair of reference points
    K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params)
        
    # added small positive diagonal to make the matrix positive definite
    sparse_covmatrix = noise**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-4
    projected_labels = K_MN@Y_data
    return sparse_covmatrix, projected_labels