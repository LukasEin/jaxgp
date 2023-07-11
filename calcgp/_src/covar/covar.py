from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap
from jax.numpy import ndarray

from ..distributions import FullPriorDistribution, SparsePriorDistribution
from ..kernels import Kernel
from ..utils import matmul_diag
from .base_covar import *


class DataMode(Enum):
    FUNC = 0
    GRAD = 1
    MIX = 2

@dataclass(frozen=True)
class Prior:
    sparse: bool = False
    mode: DataMode = DataMode.MIX

    def __call__(self) -> Callable:
        if self.sparse:
            if self.mode == DataMode.FUNC:
                return fitc_cov_func
            elif self.mode == DataMode.GRAD:
                return fitc_cov_grad
            elif self.mode == DataMode.MIX:
                return fitc_cov_mix
        else:
            if self.mode == DataMode.FUNC:
                return full_cov_func
            elif self.mode == DataMode.GRAD:
                return full_cov_grad
            elif self.mode == DataMode.MIX:
                return full_cov_mix



def full_cov_func(X_data: ndarray, Y_data: ndarray, kernel: Kernel, kernel_params: ndarray, noise: ndarray) -> FullPriorDistribution:
    K_NN = CovMatrixFF(X_data, X_data, kernel, kernel_params)
    
    return _full_cov_base(X_data, Y_data, K_NN, noise)



def full_cov_grad(X_data: ndarray, Y_data: ndarray, kernel: Kernel, kernel_params: ndarray, noise: ndarray) -> FullPriorDistribution:
    K_NN = CovMatrixGG(X_data, X_data, kernel, kernel_params)
    
    return _full_cov_base(X_data, Y_data, K_NN, noise)



def full_cov_mix(X_data: Tuple[ndarray, ndarray], Y_data: ndarray, kernel: Kernel, kernel_params: ndarray, noise: float) -> FullPriorDistribution:
    '''Calculates the full gaussian prior over the input samples in X_data.

    Parameters
    ----------
    X_data : Tuple[ndarray, ndarray]
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
    FullPriorDistribution
        Gaussian prior for the full GPR
    '''
    # Build the full covariance Matrix between all datapoints in X_data depending on if they   
    # represent function evaluations or derivative evaluations
    KF = CovMatrixFF(X_data[0], X_data[0], kernel, kernel_params)
    KD = CovMatrixFG(X_data[0], X_data[1], kernel, kernel_params)
    KDD = CovMatrixGG(X_data[1], X_data[1], kernel, kernel_params)

    K_NN = jnp.vstack((jnp.hstack((KF,KD)), 
                       jnp.hstack((KD.T,KDD))))
    
    return _full_cov_base(X_data, Y_data, K_NN, noise)



def _full_cov_base(X_data, Y_data, K_NN, noise):
    diag = jnp.diag_indices(len(K_NN))
    K_NN = K_NN.at[diag].add(noise**2)

    K_NN = jsp.linalg.cholesky(K_NN)
    return FullPriorDistribution(X_data, Y_data, K_NN)



def fitc_cov_func(X_data: ndarray, Y_data: ndarray, X_ref: ndarray, kernel: Kernel, kernel_params: ndarray, noise: float) -> SparsePriorDistribution:
    # calculates the covariance between the training points and the reference points
    K_MN = CovMatrixFF(X_ref, X_data, kernel, kernel_params)

    # diag(K_NN)
    K_NN_diag = vmap(lambda v: kernel.eval(v, v, kernel_params), in_axes=(0))(X_data)

    return _fitc_cov_base(Y_data, K_MN, K_NN_diag, X_ref, kernel, kernel_params, noise)



def fitc_cov_grad(X_data: ndarray, Y_data: ndarray, X_ref: ndarray, kernel: Kernel, kernel_params: ndarray, noise: float) -> SparsePriorDistribution:
    # calculates the covariance between the training points and the reference points
    K_MN = CovMatrixFG(X_ref, X_data, kernel, kernel_params)

    # diag(K_NN)
    K_NN_diag = jnp.ravel(vmap(lambda v: jnp.diag(kernel.jac(v, v, kernel_params)), in_axes=(0))(X_data))

    return _fitc_cov_base(Y_data, K_MN, K_NN_diag, X_ref, kernel, kernel_params, noise)   



def fitc_cov_mix(X_data: Tuple[ndarray, ndarray], Y_data: ndarray, X_ref: ndarray, kernel: Kernel, kernel_params: ndarray, noise: float) -> SparsePriorDistribution:
    '''Calculates the sparse gaussian prior over the input samples in X_data, projected onto the reference points X_ref.

    Parameters
    ----------
    X_data : Tuple[ndarray, ndarray]
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
    SparsePriorDistribution
        gaussian prior for the sparse GPR
    '''
    # calculates the covariance between the training points and the reference points
    KF = CovMatrixFF(X_ref, X_data[0], kernel, kernel_params)
    KD = CovMatrixFG(X_ref, X_data[1], kernel, kernel_params)
    K_MN = jnp.hstack((KF,KD))

    # diag(K_NN)
    func = vmap(lambda v: kernel.eval(v, v, kernel_params), in_axes=(0))(X_data[0])
    der = jnp.ravel(vmap(lambda v: jnp.diag(kernel.jac(v, v, kernel_params)), in_axes=(0))(X_data[1]))
    K_NN_diag = jnp.hstack((func, der))

    return _fitc_cov_base(Y_data, K_MN, K_NN_diag, X_ref, kernel, kernel_params, noise)



def _fitc_cov_base(Y_data, K_MN, K_NN_diag, X_ref, kernel, kernel_params, noise):
    noise_ref = 1e-2

    # calculates the covariance between each pair of reference points
    K_ref = CovMatrixFF(X_ref, X_ref, kernel, kernel_params)
    diag = jnp.diag_indices(len(K_ref))
    K_ref = K_ref.at[diag].add(noise_ref**2)

    # upper cholesky factor of K_ref || U_ref.T@U_ref = K_ref
    U_ref = jsp.linalg.cholesky(K_ref)

    # V is solution to U_ref.T@V = K_MN
    V = jsp.linalg.solve_triangular(U_ref.T, K_MN, lower=True)

    # diag(K_NN)
    # diag(Q_NN)
    Q_NN_diag = vmap(lambda x: x.T@x, in_axes=(1,))(V)    
    # diag(K_NN) + noise**2 - diag(Q_NN)
    fitc_diag = K_NN_diag + noise**2 - Q_NN_diag

    # (1 / sqrt(fitc_diag))@V.T
    V_scaled = matmul_diag(1 / jnp.sqrt(fitc_diag), V.T).T

    U_inv = jsp.linalg.cholesky((V_scaled@V_scaled.T).at[diag].add(1.0))

    projected_label = jsp.linalg.solve_triangular(U_inv.T, V@(Y_data / fitc_diag), lower=True)

    return SparsePriorDistribution(Y_data, X_ref, U_ref, U_inv, fitc_diag, projected_label)