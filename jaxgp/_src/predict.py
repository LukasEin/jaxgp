from typing import NamedTuple, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax.numpy import ndarray

from .containers import FullPrior, Posterior, SparsePrior
from .covar import CovMatrixFD, CovMatrixFF, CovVectorID
from .kernels import Kernel
from .utils import inner_map

def full_predict_grad(X: ndarray, covar_module: FullPrior, X_split: Tuple[ndarray, ndarray], kernel: Kernel, kernel_params: ndarray) -> Posterior:
    full_vectors = CovMatrixFD(X, X_split, kernel, kernel_params)

    means = full_vectors@jsp.linalg.cho_solve((covar_module.k_nn, False),covar_module.y_data)

    K_XX = CovVectorID(X, kernel, kernel_params)
    
    K_XNNX = inner_map(covar_module.k_nn, full_vectors)       
    stds = jnp.sqrt(K_XX - K_XNNX)
    
    return Posterior(means, stds)

def full_predict(X: ndarray, covar_module: FullPrior, X_split: Tuple[ndarray, ndarray], kernel: Kernel, kernel_params: ndarray) -> Posterior:
    '''Calculates the posterior mean and std for each point in X given prior information of the full GPR model

    Parameters
    ----------
    X : ndarray
        shape (n_points, n_features). For each point the posterior mean and std are calculated
    covar_module : FullPrior
        prior distribution of the full GPR model
    X_split : Tuple[ndarray, ndarray]
        Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    Posterior
        Posterior means and stds, [mean(x), std(x) for x in X]
    '''
    function_vectors = CovMatrixFF(X, X_split[0], kernel, kernel_params)
    derivative_vectors = CovMatrixFD(X, X_split[1], kernel, kernel_params)
    full_vectors = jnp.hstack((function_vectors, derivative_vectors))

    means = full_vectors@jsp.linalg.cho_solve((covar_module.k_nn, False),covar_module.y_data)

    K_XX = CovVectorID(X, kernel, kernel_params)
    
    K_XNNX = inner_map(covar_module.k_nn, full_vectors)       
    stds = jnp.sqrt(K_XX - K_XNNX)
    
    return Posterior(means, stds)

def sparse_predict(X: ndarray, covar_module: SparsePrior, X_ref: ndarray, kernel: Kernel, kernel_params: ndarray) -> Posterior:
    '''Calculates the posterior mean and std for each point in X given prior information of the sparse GPR model

    Parameters
    ----------
    X : ndarray
        shape (n_points, n_features). For each point the posterior mean and std are calculated
    covar_module : SparsePrior
        prior distribution of the sparse GPR model
    X_ref : ndarray
        shape (n_reference_points, n_dims). Reference points onto which the full data is projected for sparsification.
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    Posterior
        Posterior means and stds, [mean(x), std(x) for x in X]
    '''
    ref_vectors = CovMatrixFF(X, X_ref, kernel, kernel_params)

    means_left = jsp.linalg.solve_triangular(covar_module.U_inv.T, jsp.linalg.solve_triangular(covar_module.U_ref.T, ref_vectors.T, lower=True), lower=True)

    means = means_left.T@covar_module.proj_labs

    K_XX = CovVectorID(X, kernel, kernel_params)

    Q_XX = inner_map(covar_module.U_ref, ref_vectors)
    K_XMMX = inner_map(covar_module.U_inv@covar_module.U_ref, ref_vectors)
    
    stds = jnp.sqrt(K_XX - Q_XX + K_XMMX) 
    
    return Posterior(means, stds)