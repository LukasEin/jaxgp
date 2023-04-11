import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import Array, jit

from typing import Tuple, Union
from functools import partial

from .utils import _CovMatrix_Kernel
from .utils import _CovMatrix_Grad


@partial(jit, static_argnums=(0))
def full_negativeLogLikelyhood(self, params: Array, Y_data: Array, X_split: list[Array]) -> float:
    '''
        for PPA the Y_data ~ N(0,[id*s**2 + K_NN])
            which is the same as for Nystrom approximation

        The negative log likelyhood estimate is calculated according 
            to this distribution.

        Everything that is unnecessary to calculate the minimum has been removed

        Formally calculates:
        log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 - K_NN

        result is multiplied with a small coefficient for numerical stability 
            of the optimizer
    '''
    # calculates the full covariance matrix
    fit_matrix, fit_vector = self.forward(params, Y_data, X_split)
    fit_vector = fit_vector.reshape(-1)

    # calculates the logdet of the full covariance matrix
    _, logdet = jnp.linalg.slogdet(fit_matrix)

    # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    mle = logdet + fit_vector@solve(fit_matrix, fit_vector, assume_a="pos")
    
    return mle / 20000.0
    
@partial(jit, static_argnums=(0,))
def full_kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, X_split: list[Array]) -> float:
    '''
        for PPA the Y_data ~ N(0,[id*s**2 + K_NN])
            which is the same as for Nystrom approximation

        The negative log likelyhood estimate is calculated according 
            to this distribution.

        Everything that is unnecessary to calculate the minimum has been removed

        Formally calculates:
        log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 - K_NN

        result is multiplied with a small coefficient for numerical stability 
            of the optimizer
    '''
    params = jnp.array((noise,) + kernel_params)
    # calculates the full covariance matrix
    fit_matrix, fit_vector = self.forward(params, Y_data, X_split)
    fit_vector = fit_vector.reshape(-1)

    # calculates the logdet of the full covariance matrix
    _, logdet = jnp.linalg.slogdet(fit_matrix)

    # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    mle = logdet + fit_vector@solve(fit_matrix, fit_vector, assume_a="pos")
    
    return mle / 20000.0

def sparse_negativeLogLikelyhood(self, params: Array, Y_data: Array, X_ref: Array, X_split: list[Array]) -> float:
    '''
        for PPA the Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN])
            which is the same as for Nystrom approximation

        The negative log likelyhood estimate is calculated according 
            to this distribution.

        Everything that is unnecessary to calculate the minimum has been removed

        Formally calculates:
        log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 + K_MN.T@K_MM**(-1)@K_MN

        result is multiplied with a small coefficient for numerical stability 
            of the optimizer
    '''
    # calculates the covariance between the data and the reference points
    K_MN = _CovMatrix_Kernel(X_ref, X_split[0], params[1:])
    for i,elem in enumerate(X_split[1:]):
        K_deriv = _CovMatrix_Grad(X_ref, elem, index=i, params=params[1:])
        K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

    # calculates the covariance between the reference points
    K_ref = _CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

    # directly calculates the logdet of the Nystrom covariance matrix
    log_matrix = jnp.eye(len(Y_data)) * (1e-6 + params[0]**2) + K_MN.T@solve(K_ref,K_MN)
    _, logdet = jnp.linalg.slogdet(log_matrix)

    # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    invert_matrix = K_ref*params[0]**2 + K_MN@K_MN.T
    mle = logdet + (Y_data@Y_data -
                    Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / params[0]**2
    
    return mle / 20000.0

@partial(jit, static_argnums=(0,))
def sparse_kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, X_ref: Array, X_split: list[Array]) -> float:
    '''
        for PPA the Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN])
            which is the same as for Nystrom approximation

        The negative log likelyhood estimate is calculated according 
            to this distribution.

        Everything that is unnecessary to calculate the minimum has been removed

        Formally calculates:
        log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 + K_MN.T@K_MM**(-1)@K_MN

        result is multiplied with a small coefficient for numerical stability 
            of the optimizer
    '''
    # calculates the covariance between the data and the reference points
    K_MN = _CovMatrix_Kernel(X_ref, X_split[0], kernel_params)
    for i,elem in enumerate(X_split[1:]):
        K_deriv = _CovMatrix_Grad(X_ref, elem, index=i, params=kernel_params)
        K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

    # calculates the covariance between the reference points
    K_ref = _CovMatrix_Kernel(X_ref, X_ref, params=kernel_params)

    # directly calculates the logdet of the Nystrom covariance matrix
    log_matrix = jnp.eye(len(Y_data)) * (1e-6 + noise**2) + K_MN.T@solve(K_ref,K_MN)
    _, logdet = jnp.linalg.slogdet(log_matrix)

    # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
    invert_matrix = K_ref*noise**2 + K_MN@K_MN.T
    mle = logdet + (Y_data@Y_data -
                    Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / noise**2
    
    return mle / 20000.0