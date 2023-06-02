from typing import Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax.numpy import ndarray

from .covar import FullCovar, SparseCovar
from .kernels import BaseKernel
from .utils import CovMatrixFD, CovMatrixFF, _CovVector_Id, inner_map


def full_predict(X: ndarray, covar_module: FullCovar, Y_data: ndarray, X_split: ndarray, kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
    '''Calculates the posterior mean and std for each point in X given prior information 
    in the form of full_covmatrix and Y_data for the full gpr model

    Parameters
    ----------
    X : ndarray
        ndarray of shape (n_points, n_features). For each point the posterior mean and std are calculated
    full_covmatrix : ndarray
        prior covariance matrix of shape (n_samples, n_samples)
    Y_data : ndarray
        ndarray of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    X_split : ndarray
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    noise : Union[ndarray, float]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        ndarray is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    Tuple[ndarray, ndarray]
        Posterior means and stds, [mean(x), std(x) for x in X]
    '''
    function_vectors = CovMatrixFF(X, X_split[0], kernel, params)
    derivative_vectors = CovMatrixFD(X, X_split[1], kernel, params)
    full_vectors = jnp.hstack((function_vectors, derivative_vectors))

    means = full_vectors@jsp.linalg.cho_solve((covar_module.k_nn, False),Y_data)

    K_XX = _CovVector_Id(X, kernel, params)
    
    K_XNNX = inner_map(covar_module.k_nn, full_vectors)       
    stds = jnp.sqrt(K_XX - K_XNNX)
    
    return means, stds

def sparse_predict(X: ndarray, covar_module: SparseCovar, X_ref: ndarray, kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
    '''Calculates the posterior mean and std for each point in X given prior information 
    in the form of sparse_covmatrix and FITC gpr model

    Parameters
    ----------
    X : ndarray
        ndarray of shape (n_points, n_features). For each point the posterior mean and std are calculated
    full_covmatrix : ndarray
        prior covariance matrix of shape (n_samples, n_samples)
    projected_labels : ndarray
        ndarray of shape (n_referencepoints,). Labels projected into the reference space via K_MN
    X_ref : ndarray
        ndarray of shape (n_referencepoints, n_features). Reference points onto which the whole input dataset is projected.
    X_split : ndarray
        List of ndarrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    noise : Union[ndarray, float]
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
        ndarray is not supported yet!!!
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    Tuple[ndarray, ndarray]
        Posterior means and stds, [mean(x), std(x) for x in X]
    '''
    ref_vectors = CovMatrixFF(X, X_ref, kernel, params)

    means_left = jsp.linalg.solve_triangular(covar_module.U_inv.T, jsp.linalg.solve_triangular(covar_module.U_ref.T, ref_vectors.T, lower=True), lower=True)

    means = means_left.T@covar_module.proj_labs

    K_XX = _CovVector_Id(X, kernel, params)

    Q_XX = inner_map(covar_module.U_ref, ref_vectors)
    K_XMMX = inner_map(covar_module.U_inv@covar_module.U_ref, ref_vectors)
    
    stds = jnp.sqrt(K_XX - Q_XX + K_XMMX) 
    
    return means, stds