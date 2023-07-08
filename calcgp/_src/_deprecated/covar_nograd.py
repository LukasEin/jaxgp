from typing import Tuple, Union

import jax.numpy as jnp
from jax.numpy import ndarray

from ..kernels import BaseKernel
from ..utils import CovMatrixFF


def full_covariance_matrix_nograd(X_data: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> ndarray:
    K_NN = CovMatrixFF(X_data, X_data, kernel, params)

    return (jnp.eye(len(K_NN)) * (noise**2 + 1e-6) + K_NN)

def sparse_covariance_matrix_nograd(X_data: ndarray, Y_data: ndarray, X_ref: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
    # calculates the covariance between the training points and the reference points
    K_MN = CovMatrixFF(X_ref, X_data, kernel, params)

    # calculates the covariance between each pair of reference points
    K_ref = CovMatrixFF(X_ref, X_ref, kernel, params)
        
    # added small positive diagonal to make the matrix positive definite
    sparse_covmatrix = noise**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-4
    projected_labels = K_MN@Y_data
    return sparse_covmatrix, projected_labels