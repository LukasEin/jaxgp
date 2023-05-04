from typing import Tuple, Union

import jax.numpy as jnp
from jax.numpy import ndarray
from jax import jit, vmap
from jax.scipy.linalg import solve

from .utils import (_build_xT_Ainv_x, _CovMatrix_Grad, _CovMatrix_Kernel,
                    _CovVector_Id)
from .kernels import BaseKernel


@jit
def full_predict(X: ndarray, full_covmatrix: ndarray, Y_data: ndarray, X_split: ndarray, kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
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
    function_vectors = _CovMatrix_Kernel(X, X_split[0], kernel, params)
    derivative_vectors = vmap(jnp.ravel,in_axes=0)(_CovMatrix_Grad(X, X_split[1], kernel, params))
    full_vectors = jnp.hstack((function_vectors, derivative_vectors))

    means = full_vectors@solve(full_covmatrix,Y_data,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)  
    temp = _build_xT_Ainv_x(full_covmatrix, full_vectors)      
    stds = jnp.sqrt(X_cov - temp)
    
    return means, stds

@jit
def sparse_predict(X: ndarray, sparse_covmatrix: ndarray, projected_labels: ndarray, X_ref: ndarray, noise: Union[float, ndarray], kernel: BaseKernel, params: ndarray) -> Tuple[ndarray, ndarray]:
    '''Calculates the posterior mean and std for each point in X given prior information 
    in the form of sparse_covmatrix and projected labels in the sparse (PPA) gpr model

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
    ref_vectors = _CovMatrix_Kernel(X, X_ref, kernel, params)

    means = ref_vectors@solve(sparse_covmatrix,projected_labels)#,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)

    K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params)

    first_temp = _build_xT_Ainv_x(K_ref + jnp.eye(len(X_ref)) * 1e-4, ref_vectors)
    second_temp = noise**2 * _build_xT_Ainv_x(sparse_covmatrix, ref_vectors)
    
    stds = jnp.sqrt(X_cov - first_temp + second_temp) 
    
    return means, stds