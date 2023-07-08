from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
from jax.numpy import ndarray

from .distributions import FullPriorDistribution, PosteriorDistribution, SparsePriorDistribution
from .covar.base_covar import *
from .kernels import Kernel
from .covar.covar import DataMode
from .utils import inner_map


@dataclass(frozen=True)
class Posterior:
    sparse: bool = False
    prior_mode: DataMode = DataMode.MIX
    posterior_mode: DataMode = DataMode.FUNC

    def __call__(self) -> Callable:
        if self.sparse:
            if self.posterior_mode == DataMode.FUNC:
                return fitc_posterior_F
            elif self.posterior_mode == DataMode.GRAD:
                return fitc_posterior_G
            elif self.posterior_mode == DataMode.MIX:
                raise NotImplementedError("Use two posteriors with DataMode.FUNC & DataMode.GRAD respectively instead!")
        else:
            if self.posterior_mode == DataMode.FUNC:
                if self.prior_mode == DataMode.FUNC:
                    return full_posterior_FF
                elif self.prior_mode == DataMode.GRAD:
                    return full_posterior_FG
                elif self.prior_mode == DataMode.MIX:
                    return full_posterior_FM
            elif self.posterior_mode == DataMode.GRAD:
                if self.prior_mode == DataMode.FUNC:
                    return full_posterior_GF
                elif self.prior_mode == DataMode.GRAD:
                    return full_posterior_GG
                elif self.prior_mode == DataMode.MIX:
                    return full_posterior_GM
            elif self.posterior_mode == DataMode.MIX:
                raise NotImplementedError("Use two posteriors with DataMode.FUNC & DataMode.GRAD respectively instead!")



def full_posterior_FF(X: ndarray, prior: FullPriorDistribution, kernel: Kernel, kernel_params: ndarray):
    full_vectors = CovMatrixFF(X, prior.x_data, kernel, kernel_params)
    K_XX = CovDiagF(X, kernel, kernel_params)

    return _full_posterior_base(full_vectors, K_XX, prior)



def full_posterior_FG(X: ndarray, prior: FullPriorDistribution, kernel: Kernel, kernel_params: ndarray):
    full_vectors = CovMatrixFG(X, prior.x_data, kernel, kernel_params)
    K_XX = CovDiagF(X, kernel, kernel_params)

    return _full_posterior_base(full_vectors, K_XX, prior)



def full_posterior_FM(X: ndarray, prior: FullPriorDistribution, kernel: Kernel, kernel_params: ndarray):
    f_vectors = CovMatrixFF(X, prior.x_data[0], kernel, kernel_params)
    g_vectors = CovMatrixFG(X, prior.x_data[1], kernel, kernel_params)
    full_vectors = jnp.hstack((f_vectors, g_vectors))
    K_XX = CovDiagF(X, kernel, kernel_params)

    return _full_posterior_base(full_vectors, K_XX, prior)



def full_posterior_GF(X: ndarray, prior: FullPriorDistribution, kernel: Kernel, kernel_params: ndarray):
    full_vectors = CovMatrixFG(prior.x_data, X, kernel, kernel_params).T
    K_XX = CovDiagG(X, kernel, kernel_params)

    return _full_posterior_base(full_vectors, K_XX, prior)



def full_posterior_GG(X: ndarray, prior: FullPriorDistribution, kernel: Kernel, kernel_params: ndarray):
    full_vectors = CovMatrixGG(X, prior.x_data, kernel, kernel_params)
    K_XX = CovDiagG(X, kernel, kernel_params)

    return _full_posterior_base(full_vectors, K_XX, prior)



def full_posterior_GM(X: ndarray, prior: FullPriorDistribution, kernel: Kernel, kernel_params: ndarray):
    f_vectors = CovMatrixFG(prior.x_data[0], X, kernel, kernel_params).T
    g_vectors = CovMatrixGG(X, prior.x_data[1], kernel, kernel_params)
    full_vectors = jnp.hstack((f_vectors, g_vectors))
    K_XX = CovDiagG(X, kernel, kernel_params)

    return _full_posterior_base(full_vectors, K_XX, prior)



def _full_posterior_base(full_vectors: ndarray, K_XX: ndarray, prior: FullPriorDistribution) -> PosteriorDistribution:
    '''Calculates the posterior mean and std for each point in X given prior information of the full GPR model

    Parameters
    ----------
    X : ndarray
        shape (n_points, n_features). For each point the posterior mean and std are calculated
    prior : FullPriorDistribution
        prior distribution of the full GPR model
    X_split : Tuple[ndarray, ndarray]
        Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    PosteriorDistribution
        PosteriorDistribution means and stds, [mean(x), std(x) for x in X]
    '''
    means = full_vectors@jsp.linalg.cho_solve((prior.k_nn, False),prior.y_data)
    
    K_XNNX = inner_map(prior.k_nn, full_vectors)       
    stds = jnp.sqrt(K_XX - K_XNNX)
    
    return PosteriorDistribution(means, stds)



def fitc_posterior_F(X: ndarray, prior: SparsePriorDistribution, kernel: Kernel, kernel_params: ndarray) -> PosteriorDistribution:
    fitc_vectors = CovMatrixFF(X, prior.x_ref, kernel, kernel_params)
    K_XX = CovDiagF(X, kernel, kernel_params)

    return _fitc_posterior_base(fitc_vectors, K_XX, prior)



def fitc_posterior_G(X: ndarray, prior: SparsePriorDistribution, kernel: Kernel, kernel_params: ndarray) -> PosteriorDistribution:
    fitc_vectors = CovMatrixFG(prior.x_ref, X, kernel, kernel_params).T
    K_XX = CovDiagG(X, kernel, kernel_params)

    return _fitc_posterior_base(fitc_vectors, K_XX, prior)



def _fitc_posterior_base(ref_vectors: ndarray, K_XX: ndarray, prior: SparsePriorDistribution) -> PosteriorDistribution:
    '''Calculates the posterior mean and std for each point in X given prior information of the sparse GPR model

    Parameters
    ----------
    X : ndarray
        shape (n_points, n_features). For each point the posterior mean and std are calculated
    prior : SparsePriorDistribution
        prior distribution of the sparse GPR model
    X_ref : ndarray
        shape (n_reference_points, n_dims). Reference points onto which the full data is projected for sparsification.
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    PosteriorDistribution
        PosteriorDistribution means and stds, [mean(x), std(x) for x in X]
    '''
    means_left = jsp.linalg.solve_triangular(prior.U_inv.T, jsp.linalg.solve_triangular(prior.U_ref.T, ref_vectors.T, lower=True), lower=True)

    means = means_left.T@prior.proj_labs

    Q_XX = inner_map(prior.U_ref, ref_vectors)
    K_XMMX = inner_map(prior.U_inv@prior.U_ref, ref_vectors)
    
    stds = jnp.sqrt(K_XX - Q_XX + K_XMMX) 
    
    return PosteriorDistribution(means, stds)