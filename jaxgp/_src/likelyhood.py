from typing import Union
from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.linalg import solve

from .covar import full_covariance_matrix
from .kernels import BaseKernel
from .utils import _CovMatrix_Grad, _CovMatrix_Kernel


@jit 
def full_kernelNegativeLogLikelyhood(kernel_params: Array, X_split: list[Array], Y_data: Array, noise: Union[Array, float], kernel: BaseKernel) -> float:
    '''Negative log Likelyhood for full GPR. Y_data ~ N(0,[id*s**2 + K_NN]).
    kernel_params are the first arguments in order to minimize this function w.r.t. those variables.

    Parameters
    ----------
    kernel_params : Array
        kernel parameters. Function can be optimized w.r.t to these parameters
    X_split : list[Array]
        List of Arrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : Array
        Array of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    noise : Union[Array, float]
        either scalar or Array of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        Array is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.

    Returns
    -------
    float
        Negative Log Likelyhood estimate for full GPR
    '''
    # calculates the full covariance matrix
    fit_matrix = full_covariance_matrix(X_split, noise, kernel, kernel_params)
    fit_vector = Y_data.reshape(-1)

    # calculates the logdet of the full covariance matrix
    _, logdet = jnp.linalg.slogdet(fit_matrix)

    # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    mle = 0.5*(logdet + fit_vector@solve(fit_matrix, fit_vector, assume_a="pos"))
    
    return mle / 20000.0

@jit
def sparse_kernelNegativeLogLikelyhood(kernel_params: Array, X_split: list[Array], Y_data: Array, X_ref: Array, noise: Union[Array, float], kernel: BaseKernel) -> float:
    '''Negative log Likelyhood for sparse GPR (PPA). Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN]) which is the same as for Nystrom approximation.
    kernel_params are the first arguments in order to minimize this function w.r.t. those variables.

    Parameters
    ----------
    kernel_params : Array
        kernel parameters. Function can be optimized w.r.t to these parameters
    X_split : list[Array]
        List of Arrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : Array
        Array of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    X_ref : Array
        Array of shape (n_referencepoints, n_features). Reference points onto which the whole input dataset is projected.
    noise : Union[Array, float]
        either scalar or Array of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        Array is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.

    Returns
    -------
    float
        Negative Log Likelyhood estimate for PPA
    '''
    # calculates the covariance between the data and the reference points
    K_MN = _CovMatrix_Kernel(X_ref, X_split[0], kernel, kernel_params)
    for i,elem in enumerate(X_split[1:]):
        K_deriv = _CovMatrix_Grad(X_ref, elem, kernel, kernel_params, index=i)
        K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

    # calculates the covariance between the reference points
    K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, kernel_params)

    # directly calculates the logdet of the Nystrom covariance matrix
    log_matrix = jnp.eye(len(Y_data)) * (1e-6 + noise**2) + K_MN.T@solve(K_ref,K_MN)
    _, logdet = jnp.linalg.slogdet(log_matrix)

    # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    invert_matrix = K_ref*noise**2 + K_MN@K_MN.T
    mle = 0.5*(logdet + (Y_data@Y_data -
                    Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / noise**2)
    
    return mle / 20000.0

# def sparse_negativeLogLikelyhood(self, params: Array, Y_data: Array, X_ref: Array, X_split: list[Array]) -> float:
#     '''
#         for PPA the Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN])
#             which is the same as for Nystrom approximation

#         The negative log likelyhood estimate is calculated according 
#             to this distribution.

#         Everything that is unnecessary to calculate the minimum has been removed

#         Formally calculates:
#         log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 + K_MN.T@K_MM**(-1)@K_MN

#         result is multiplied with a small coefficient for numerical stability 
#             of the optimizer
#     '''
#     # calculates the covariance between the data and the reference points
#     K_MN = _CovMatrix_Kernel(X_ref, X_split[0], params[1:])
#     for i,elem in enumerate(X_split[1:]):
#         K_deriv = _CovMatrix_Grad(X_ref, elem, index=i, params=params[1:])
#         K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

#     # calculates the covariance between the reference points
#     K_ref = _CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

#     # directly calculates the logdet of the Nystrom covariance matrix
#     log_matrix = jnp.eye(len(Y_data)) * (1e-6 + params[0]**2) + K_MN.T@solve(K_ref,K_MN)
#     _, logdet = jnp.linalg.slogdet(log_matrix)

#     # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
#     invert_matrix = K_ref*params[0]**2 + K_MN@K_MN.T
#     mle = logdet + (Y_data@Y_data -
#                     Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / params[0]**2
    
#     return mle / 20000.0

# def full_negativeLogLikelyhood(params: Array, X_split: list[Array], Y_data: Array, noise: Union[float, Array], kernel: BaseKernel) -> float:
#     '''
#         for PPA the Y_data ~ N(0,[id*s**2 + K_NN])
#             which is the same as for Nystrom approximation

#         The negative log likelyhood estimate is calculated according 
#             to this distribution.

#         Everything that is unnecessary to calculate the minimum has been removed

#         Formally calculates:
#         log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 - K_NN

#         result is multiplied with a small coefficient for numerical stability 
#             of the optimizer
#     '''
#     # calculates the full covariance matrix
#     fit_matrix = full_covariance_matrix(X_split, noise, kernel, params)
#     fit_vector = Y_data.reshape(-1)

#     # calculates the logdet of the full covariance matrix
#     _, logdet = jnp.linalg.slogdet(fit_matrix)

#     # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
#     mle = logdet + fit_vector@solve(fit_matrix, fit_vector, assume_a="pos")
    
#     return mle / 20000.0