from typing import Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax.numpy import ndarray

from .covar import full_covariance_matrix, sparse_covariance_matrix
from .kernels import Kernel


def full_NLML(X_split: Tuple[ndarray, ndarray], Y_data: ndarray, kernel: Kernel, kernel_params: ndarray, noise: float) -> float:
    '''Negative log marginal likelihood for the full GPR

    Parameters
    ----------
    X_split : Tuple[ndarray, ndarray]
        Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
    Y_data : ndarray
        shape (n_function_evals + n_gradient_evals, ). Input labels representing noisy function evaluations
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters
    noise : float
        describes the noise present in the given input labels.

    Returns
    -------
    float
    '''
    covar_module = full_covariance_matrix(X_split, Y_data, kernel, kernel_params, noise)

    # logdet calculattion
    K_NN_diag = jnp.diag(covar_module.k_nn)
    logdet = 2*jnp.sum(jnp.log(K_NN_diag))

    # Fit calculation
    fit = Y_data.T@jsp.linalg.cho_solve((covar_module.k_nn, False), Y_data)

    nlle = 0.5*(logdet + fit + len(Y_data)*jnp.log(2*jnp.pi))
    
    return nlle / len(Y_data)

def sparse_NLML(X_split: Tuple[ndarray, ndarray], Y_data: ndarray, X_ref: ndarray, kernel: Kernel, kernel_params: ndarray, noise: float) -> float:
    '''Negative log marginal likelihood for the sparse GPR

    Parameters
    ----------
    X_split : Tuple[ndarray, ndarray]
        Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
    Y_data : ndarray
        shape (n_function_evals + n_gradient_evals, ). Input labels representing noisy function evaluations
    X_ref : ndarray
        shape (n_referencepoints, n_dims). Reference points onto which the whole input dataset is projected.
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters
    noise : float
        describes the noise present in the given input labels.

    Returns
    -------
    float
    '''
    covar_module = sparse_covariance_matrix(X_split, Y_data, X_ref, kernel, kernel_params, noise)

    # Logdet calculations
    U_inv_diag = jnp.diag(covar_module.U_inv)
    logdet_K_inv = 2*jnp.sum(jnp.log(U_inv_diag))
    logdet_fitc = jnp.sum(jnp.log(covar_module.diag))

    # Fit calculation
    Y_scaled = Y_data / jnp.sqrt(covar_module.diag)
    fit = Y_scaled.T@Y_scaled - covar_module.proj_labs.T@covar_module.proj_labs

    nlle = 0.5*(logdet_fitc + logdet_K_inv + fit + len(Y_data)*jnp.log(2*jnp.pi))
    
    return nlle / len(Y_data)