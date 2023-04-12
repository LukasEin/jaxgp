import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import Array, jit

from typing import Tuple

from .utils import _CovMatrix_Kernel
from .utils import _CovMatrix_Grad
from .utils import _CovVector_Id
from .utils import _build_xT_Ainv_x

@jit
def full_predict(X: Array, full_covmatrix: Array, Y_data: Array, X_split: Array, kernel, params: Array) -> Tuple[Array, Array]:
    '''Calculates the posterior mean and std for each point in X 
    given prior information in the form of fit

    Parameters
    ----------
    X : Array
        Array of shape (n_points, n_features). For each point the posterior mean and std are calculated
    full_covmatrix : Array
        prior covariance matrix of shape (n_samples, n_samples)
    Y_data : Array
        Array of shape (n_samples,) s.t. n_samples = sum(n_samples_i) in X_split. Corresponding labels to the samples in X_split
    X_split : Array
        List of Arrays: [function_evals(n_samples_f, n_features), dx1_evals(n_samples_dx1, n_features), ..., dxn_featrues_evals(n_samples_dxn_features, n_features)]
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        kernel parameters

    Returns
    -------
    Tuple[Array, Array]
        Posterior means and stds for each point in X
    '''
    full_vectors = _CovMatrix_Kernel(X, X_split[0], kernel, params=params)
    for i,elem in enumerate(X_split[1:]):
        deriv_vectors = _CovMatrix_Grad(X, elem, kernel, params=params, index=i)
        full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

    means = full_vectors@solve(full_covmatrix,Y_data,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)  
    temp = _build_xT_Ainv_x(full_covmatrix, full_vectors)      
    stds = jnp.sqrt(X_cov - temp) # no noise term in the variance
    
    return means, stds

@jit
def sparse_predict(X: Array, fitmatrix: Array, fitvector: Array, X_ref: Array, noise, kernel, params: Array) -> Tuple[Array, Array]:
    '''
        calculates the posterior mean and std for all points in X
    '''
    # calculates the covariance between the test points and the reference points
    ref_vectors = _CovMatrix_Kernel(X, X_ref, kernel, params)

    means = ref_vectors@solve(fitmatrix,fitvector)#,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)

    K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params)

    first_temp = _build_xT_Ainv_x(K_ref + jnp.eye(len(X_ref)) * 1e-4, ref_vectors)
    second_temp = noise**2 * _build_xT_Ainv_x(fitmatrix, ref_vectors)
    
    stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
    
    return means, stds