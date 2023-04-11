import jax.numpy as jnp
from jax import vmap, jit, Array
from jax.scipy.linalg import solve
from functools import partial
from .kernels import BaseKernel

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
