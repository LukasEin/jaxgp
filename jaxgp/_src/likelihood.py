from typing import Union

import jax.numpy as jnp
from jax.numpy import ndarray
import jax.scipy as jsp

from .covar import full_covariance_matrix, sparse_covariance_matrix
from .kernels import BaseKernel


def full_NLML(X_split: list[ndarray], Y_data: ndarray, kernel: BaseKernel, kernel_params: ndarray, noise: Union[ndarray, float]) -> float:
    '''Negative log marginal likelihood for full GPR. Y_data ~ N(0,[id*s**2 + K_NN]).

    Parameters
    ----------
    X_split : list[ndarray]
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : ndarray
        ndarray of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters. Function can be optimized w.r.t to these parameters
    noise : Union[ndarray, float]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        ndarray is not supported yet!!!

    Returns
    -------
    float
        Negative log marginal likelihood for full GPR
    '''
    covar_module = full_covariance_matrix(X_split, Y_data, noise, kernel, kernel_params)

    # logdet calculattion
    K_NN_diag = jnp.diag(covar_module.k_nn)
    logdet = 2*jnp.sum(jnp.log(K_NN_diag))

    # Fit calculation
    fit = Y_data.T@jsp.linalg.cho_solve((covar_module.k_nn, False), Y_data)

    nlle = 0.5*(logdet + fit + len(Y_data)*jnp.log(2*jnp.pi))
    
    return nlle / len(Y_data)

def sparse_NLML(X_split: list[ndarray], Y_data: ndarray, kernel: BaseKernel, X_ref: ndarray, kernel_params: ndarray, noise: Union[ndarray, float]) -> float:
    '''Negative log marginal likelyhood for sparse GPR (FITC). Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN]) which is the same as for Nystrom approximation.

    Parameters
    ----------
    X_split : list[ndarray]
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    Y_data : ndarray
        ndarray of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    X_ref : ndarray
        ndarray of shape (n_referencepoints, n_features). Reference points onto which the whole input dataset is projected.
    kernel_params : ndarray
        kernel parameters. Function can be optimized w.r.t to these parameters
    noise : Union[ndarray, float]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        ndarray is not supported yet!!!

    Returns
    -------
    float
        Negative log marginal likelyhood for FITC
    '''
    covar_module = sparse_covariance_matrix(X_split, Y_data, X_ref, noise, kernel, kernel_params)

    # Logdet calculations
    U_inv_diag = jnp.diag(covar_module.U_inv)
    logdet_K_inv = 2*jnp.sum(jnp.log(U_inv_diag))
    logdet_fitc = jnp.sum(jnp.log(covar_module.diag))

    # Fit calculation
    Y_scaled = Y_data / jnp.sqrt(covar_module.diag)
    fit = Y_scaled.T@Y_scaled - covar_module.proj_labs.T@covar_module.proj_labs

    nlle = 0.5*(logdet_fitc + logdet_K_inv + fit + len(Y_data)*jnp.log(2*jnp.pi))
    
    return nlle / len(Y_data)