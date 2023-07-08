from typing import Union

import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy import ndarray
from jax.scipy.linalg import solve
import jax.scipy as jsp

from .old_covar import full_covariance_matrix, sparse_covariance_matrix
from ..kernels import BaseKernel
from ..utils import CovMatrixFD, CovMatrixFF
from .covar_nograd import full_covariance_matrix_nograd



def full_kernelNegativeLogLikelyhood_nograd(kernel_params: ndarray, X_data: ndarray, Y_data: ndarray, noise: Union[ndarray, float], kernel: BaseKernel) -> float:
    '''Negative log Likelyhood for full GPR. Y_data ~ N(0,[id*s**2 + K_NN]).
    kernel_params are the first arguments in order to minimize this function w.r.t. those variables.

    Parameters
    ----------
    kernel_params : ndarray
        kernel parameters. Function can be optimized w.r.t to these parameters
    X_split : list[ndarray]
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : ndarray
        ndarray of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    noise : Union[ndarray, float]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        ndarray is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.

    Returns
    -------
    float
        Negative Log Likelyhood estimate for full GPR
    '''
    # calculates the full covariance matrix
    fit_matrix = full_covariance_matrix_nograd(X_data, noise, kernel, kernel_params)
    fit_vector = Y_data.reshape(-1)

    # calculates the logdet of the full covariance matrix
    _, logdet = jnp.linalg.slogdet(fit_matrix)

    # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    nlle = 0.5*(logdet + 
               fit_vector@solve(fit_matrix, fit_vector, assume_a="pos") + 
               len(fit_vector)*jnp.log(2*jnp.pi))
    
    return nlle

def sparse_kernelNegativeLogLikelyhood_nograd(kernel_params: ndarray, X_data: ndarray, Y_data: ndarray, X_ref: ndarray, noise: Union[ndarray, float], kernel: BaseKernel) -> float:
    # calculates the covariance between the data and the reference points
    K_MN = CovMatrixFF(X_ref, X_data, kernel, kernel_params)

    # calculates the covariance between the reference points
    K_ref = CovMatrixFF(X_ref, X_ref, kernel, kernel_params)

    # directly calculates the logdet of the Nystrom covariance matrix
    log_matrix = jnp.eye(len(Y_data)) * (1e-6 + noise**2) + K_MN.T@solve(K_ref,K_MN)
    _, logdet = jnp.linalg.slogdet(log_matrix)

    # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    invert_matrix = K_ref*noise**2 + K_MN@K_MN.T
    nlle = 0.5*(logdet + 
               (Y_data@Y_data - Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / noise**2 + 
               len(Y_data)*jnp.log(2*jnp.pi))
    
    return nlle