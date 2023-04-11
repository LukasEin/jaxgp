import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import Array, jit

from typing import Tuple

from .utils import _CovMatrix_Kernel
from .utils import _CovMatrix_Grad
from .utils import _CovVector_Id
from .utils import _build_xT_Ainv_x

@jit
def full_meanstd(X: Array, fitmatrix: Array, fitvector: Array, X_split: Array, kernel, params: Array) -> Tuple[Array, Array]:
        '''
            calculates the posterior mean and std for all points in X
        '''
        full_vectors = _CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = _CovMatrix_Grad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector,assume_a="pos")

        X_cov = _CovVector_Id(X, params[1:])  
        temp = _build_xT_Ainv_x(fitmatrix, full_vectors)      
        stds = jnp.sqrt(X_cov - temp) # no noise term in the variance
        
        return means, stds

@jit
def sparse_meanstd(X: Array, fitmatrix: Array, fitvector: Array, X_ref: Array, noise, kernel, params: Array, X_split: list[Array]) -> Tuple[Array, Array]:
    '''
        calculates the posterior mean and std for all points in X
    '''
    # calculates the caovariance between the test points and the reference points
    ref_vectors = _CovMatrix_Kernel(X, X_ref, kernel, params)

    means = ref_vectors@solve(fitmatrix,fitvector)#,assume_a="pos")

    X_cov = _CovVector_Id(X, kernel, params)

    K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params=params)

    first_temp = _build_xT_Ainv_x(K_ref + jnp.eye(len(X_ref)) * 1e-4, ref_vectors)
    second_temp = noise**2 * _build_xT_Ainv_x(fitmatrix, ref_vectors)
    
    stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
    
    return means, stds