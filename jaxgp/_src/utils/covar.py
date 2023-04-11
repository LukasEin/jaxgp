import jax.numpy as jnp
from jax import Array, jit

from typing import Tuple

from .utils import _CovMatrix_Kernel
from .utils import _CovMatrix_Grad
from .utils import _CovMatrix_Hess

from ..kernels import BaseKernel

@jit
def full_covariance_matrix(X_split: list[Array], noise, kernel: BaseKernel, params: Array) -> Array:
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
def sparse_covariance_matrix(X_split, Y_data, X_ref, noise, kernel, params) -> Tuple[Array, Array]:
        # calculates the covariance between the training points and the reference points
        K_MN = _CovMatrix_Kernel(X_ref, X_split[0], kernel, params)
        for i,elem in enumerate(X_split[1:]):
            K_deriv = _CovMatrix_Grad(X_ref, elem, kernel, params=params, index=i)
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        # calculates the covariance between each pair of reference points
        K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params=params)
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = noise**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-4
        fit_vector = K_MN@Y_data
        return fit_matrix, fit_vector