from typing import Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax.numpy import ndarray
from jax.scipy.linalg import solve

from ..covar import FullCovar, SparseCovar
from ..kernels import BaseKernel
from ..utils import CovMatrixFD, CovMatrixFF, _build_xT_Ainv_x, _CovVector_Id


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
    ref_vectors = CovMatrixFF(X, X_ref, kernel, params)

    means = ref_vectors@solve(sparse_covmatrix,projected_labels)#,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)

    K_ref = CovMatrixFF(X_ref, X_ref, kernel, params)

    helper = vmap(lambda A, x: x.T@solve(A,x), in_axes=(None, 0))
    first_temp = helper(K_ref + jnp.eye(len(X_ref)) * 1e-4, ref_vectors)
    second_temp = helper(sparse_covmatrix, ref_vectors)
    
    stds = jnp.sqrt(X_cov - first_temp + second_temp) 
    
    return means, stds