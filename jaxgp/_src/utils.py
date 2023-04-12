from functools import partial

from jax import Array, jit, vmap
from jax.scipy.linalg import solve

from .kernels import BaseKernel


@jit
@partial(vmap, in_axes=(None, 0))
def _build_xT_Ainv_x(A: Array, X: Array) -> Array:
    '''
        X.shape = (N,M)
        A.shape = (M,M)

        output.shape = (N,)

        Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
    '''
    return X.T@solve(A,X,assume_a="pos")

@jit
def _CovVector_Id(X: Array, kernel: BaseKernel, params: Array) -> Array:
    '''
        X1.shape = (N, n_features)

        output.shape = (N,)

        Builds a vector of the covariance of all X[:,i] with them selves.
    '''
    func = lambda v: kernel.eval(v, v, params)
    func = vmap(func, in_axes=(0))
    return func(X)

@jit
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

@jit
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

@jit
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