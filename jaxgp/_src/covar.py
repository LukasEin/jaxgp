from typing import Tuple, Union

import jax.numpy as jnp
from jax import Array, jit

from .kernels import BaseKernel
from .utils import _CovMatrix_Grad, _CovMatrix_Hess, _CovMatrix_Kernel


@jit
def full_covariance_matrix(X_split: list[Array], noise: Union[float, Array], kernel: BaseKernel, params: Array) -> Array:
    '''Calculates the full covariance matrix over the input samples in X_split.

    Parameters
    ----------
    X_split : list[Array]
        List of Arrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    noise : Union[float, Array]
        either scalar or Array of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split.
        Array is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        Kernel parameters

    Returns
    -------
    Array
        full covariance matrix
    '''
    # Build the full covariance Matrix between all datapoints in X_data depending on if they   
    # represent function evaluations or derivative evaluations
    K_NN = _CovMatrix_Kernel(X_split[0], X_split[0], kernel, params=params)
    for i,elem in enumerate(X_split[1:]):
        K_mix = _CovMatrix_Grad(X_split[0], elem, kernel, params=params, index=i)
        K_NN = jnp.concatenate((K_NN,K_mix),axis=1)
    
    for i,elem_1 in enumerate(X_split[1:]):
        K_mix = _CovMatrix_Grad(X_split[0], elem_1, kernel, params=params, index=i)

        for j,elem_2 in enumerate(X_split[1:]):
            K_derivs = _CovMatrix_Hess(elem_1, elem_2, kernel, params=params, index_1=i, index_2=j)
            K_mix = jnp.concatenate((K_mix,K_derivs),axis=0)

        K_NN = jnp.concatenate((K_NN,K_mix.T),axis=0)

    # additional small diagonal element added for 
    # numerical stability of the inversion and determinant
    return (jnp.eye(len(K_NN)) * (noise**2 + 1e-6) + K_NN)

@jit
def sparse_covariance_matrix(X_split: list[Array], Y_data: Array, X_ref: Array, noise: Union[float, Array], kernel: BaseKernel, params: Array) -> Tuple[Array, Array]:
    '''Calculates the sparse covariance matrix over the input samples in X_split 
    and the projected input labels in Y_data according to the Projected Process Approximation.

    Parameters
    ----------
    X_split : list[Array]
        List of Arrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : Array
        Array of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    X_ref : Array
        Array of shape (n_referencepoints, n_features). Reference points onto which the whole input dataset is projected.
    noise : Union[float, Array]
        either scalar or Array of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        Array is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        Kernel parameters

    Returns
    -------
    Tuple[Array, Array]
        sparse (PPA) covariance matrix, projected input labels
    '''
    # calculates the covariance between the training points and the reference points
    K_MN = _CovMatrix_Kernel(X_ref, X_split[0], kernel, params)
    for i,elem in enumerate(X_split[1:]):
        K_deriv = _CovMatrix_Grad(X_ref, elem, kernel, params=params, index=i)
        K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

    # calculates the covariance between each pair of reference points
    K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params=params)
        
    # added small positive diagonal to make the matrix positive definite
    sparse_covmatrix = noise**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-4
    projected_labels = K_MN@Y_data
    return sparse_covmatrix, projected_labels