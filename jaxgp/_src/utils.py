import jax.numpy as jnp
from jax import vmap, jit, Array
from jax.scipy.linalg import solve
from functools import partial
from .kernels import BaseKernel

from typing import Tuple

@partial(vmap, in_axes=(None, 0))
def _build_xT_Ainv_x(A: Array, X: Array) -> Array:
    '''
        X.shape = (N,M)
        A.shape = (M,M)

        output.shape = (N,)

        Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
    '''
    return X.T@solve(A,X,assume_a="pos")

def _CovVector_Id(X: Array, kernel: BaseKernel, params: Array) -> Array:
    '''
        X1.shape = (N, n_features)

        output.shape = (N,)

        Builds a vector of the covariance of all X[:,i] with them selves.
    '''
    func = lambda v: kernel.eval(v, v, params)
    func = vmap(func, in_axes=(0))
    return func(X)

def _CovMatrix_Kernel(X1: Array, X2: Array, kernel: BaseKernel, params: Array) -> Array:
    '''
        X1.shape = (N1, n_features)
        X2.shape = (N2, n_features)

        output.shape = (N1, N2)

        Builds the covariance matrix between the elements of X1 and X2
        based on inputs representing values of the target function.
    '''
    func = lambda v1, v2: kernel.eval(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

def _CovMatrix_Grad(X1: Array, X2: Array, kernel: BaseKernel, params: Array, index: int) -> Array:
    '''
        X1.shape = (N1, n_features)
        X2.shape = (N2, n_features)
        
        index in range(0, n_features - 1), derivative of the kernel
            is taken w.r.t. to X_2[:,index].

        output.shape = (N1, N2)

        Builds the covariance matrix between the elements of X1 and X2
        based on X1 representing values of the target function and X2
        representing derivative values of the target function.
    '''
    func = lambda v1, v2: kernel.grad2(v1, v2, index, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

def _CovMatrix_Hess(X1: Array, X2: Array, kernel: BaseKernel, params: Array, index_1: int, index_2: int) -> Array:
    '''
        X1.shape = (N1, n_features)
        X2.shape = (N2, n_features)
        
        index_i in range(0, n_features - 1), double derivative of the 
            kernel is taken w.r.t. to X1[:,index_1] and X2[:,index_2].

        output.shape = (N1, N2)

        Builds the covariance matrix between the elements of X1 and X2
        based on X1 and X2 representing derivative values of the target 
        function.
    '''
    func = lambda v1, v2: kernel.jac(v1, v2, index_1, index_2, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

@jit
def full_covariance_matrix(X_split: list[Array], noise, kernel: BaseKernel, params: Array) -> Array:
    # Build the full covariance Matrix between all datapoints in X_data depending on if they   
    # represent function evaluations or derivative evaluations
    K_NN = _CovMatrix_Kernel(X_split[0], X_split[0], kernel, params=params)
    for i,elem in enumerate(X_split[1:]):
        K_mix = _CovMatrix_Grad(X_split[0], elem, kernel, params=params, index=i)
        K_NN = jnp.concatenate((K_NN,K_mix),axis=1)
    
    for i,elem_1 in enumerate(X_split[1:]):
        K_mix = _CovMatrix_Grad(X_split[0], elem_1, kernel, params=params, index=i)

        for j,elem_2 in enumerate(X_split[1:]):
            K_derivs = _CovMatrix_Hess(elem_1, elem_2, kernel, params=params, index_1=i, index_2=j)
            K_mix = jnp.concatenate((K_mix,K_derivs),axis=0)

        K_NN = jnp.concatenate((K_NN,K_mix.T),axis=0)

    # additional small diagonal element added for 
    # numerical stability of the inversion and determinant
    return (jnp.eye(len(K_NN)) * (noise**2 + 1e-6) + K_NN)

@jit
def sparse_covariance_matrix(X_split, Y_data, X_ref, noise, kernel, params) -> Tuple[Array, Array]:
        # calculates the covariance between the training points and the reference points
        K_MN = _CovMatrix_Kernel(X_ref, X_split[0], kernel, params)
        for i,elem in enumerate(X_split[1:]):
            K_deriv = _CovMatrix_Grad(X_ref, elem, kernel, params=params, index=i)
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        # calculates the covariance between each pair of reference points
        K_ref = _CovMatrix_Kernel(X_ref, X_ref, kernel, params=params)
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = noise**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-4
        fit_vector = K_MN@Y_data
        return fit_matrix, fit_vector

@jit
def _meanstd(X: Array, fitmatrix: Array, fitvector: Array, X_split: Array, kernel, params: Array) -> Tuple[Array, Array]:
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

