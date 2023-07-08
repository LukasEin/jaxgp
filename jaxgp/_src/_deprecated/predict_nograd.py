from typing import Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax.numpy import ndarray
from jax.scipy.linalg import solve

from .old_covar import FullCovar, SparseCovar
from ..kernels import BaseKernel
from ..utils import CovMatrixFD, CovMatrixFF, _build_xT_Ainv_x, _CovVector_Id


def full_predict_nograd(X: ndarray, full_covmatrix_nograd: ndarray, Y_data: ndarray, X_data: ndarray, kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
    full_vectors = CovMatrixFF(X, X_data, kernel, params)

    means = full_vectors@solve(full_covmatrix_nograd,Y_data,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)  
    temp = _build_xT_Ainv_x(full_covmatrix_nograd, full_vectors)      
    stds = jnp.sqrt(X_cov - temp)

    return means, stds

def sparse_predict_nograd(X: ndarray, sparse_covmatrix: ndarray, projected_labels: ndarray, X_ref: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
    ref_vectors = CovMatrixFF(X, X_ref, kernel, params)

    means = ref_vectors@solve(sparse_covmatrix,projected_labels)#,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)

    K_ref = CovMatrixFF(X_ref, X_ref, kernel, params)

    first_temp = _build_xT_Ainv_x(K_ref + jnp.eye(len(X_ref)) * 1e-4, ref_vectors)
    second_temp = noise**2 * _build_xT_Ainv_x(sparse_covmatrix, ref_vectors)
    
    stds = jnp.sqrt(X_cov - first_temp + second_temp) 
    
    return means, stds