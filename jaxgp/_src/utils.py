from functools import partial

from jax import Array, jit, vmap
from jax.scipy.linalg import solve

from .kernels import BaseKernel


@jit
@partial(vmap, in_axes=(None, 0))
def _build_xT_Ainv_x(A: Array, X: Array) -> Array:
    ''' Calculates X.T @ A_inv @ X for each of the M arrays in X

    Parameters
    ----------
    A : Array
        positive definite matrix of shape (N, N)
    X : Array
        M arrays of shape (N, )

    Returns
    -------
    Array
        shape (M, ), [x.T @ A_inv @ x for x in X]
    '''
    return X.T@solve(A,X,assume_a="pos")

@jit
def _CovVector_Id(X: Array, kernel: BaseKernel, params: Array) -> Array:
    '''Calculates the covariance of each point in X with itself

    Parameters
    ----------
    X : Array
        array of shape (N, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        kernel parameters

    Returns
    -------
    Array
        shape (N, ), [K(x, x) for x in X]
    '''
    func = lambda v: kernel.eval(v, v, params)
    func = vmap(func, in_axes=(0))
    return func(X)

@jit
def _CovMatrix_Kernel(X1: Array, X2: Array, kernel: BaseKernel, params: Array) -> Array:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on inputs representing values of the target function.

    Parameters
    ----------
    X1 : Array
        shape (N1, n_features)
    X2 : Array
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        kernel parameters

    Returns
    -------
    Array
        shape (N1, N2), [K(x1, x2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.eval(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

@jit
def _CovMatrix_Grad(X1: Array, X2: Array, kernel: BaseKernel, params: Array, index: int) -> Array:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 representing values of the target function and X2
    representing derivative values of the target function.

    Parameters
    ----------
    X1 : Array
        shape (N1, n_features)
    X2 : Array
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        kernel parameters
    index : int
        derivative of the kernel is calculted w.r.t. to X2[index]

    Returns
    -------
    Array
        shape (N1, N2), [dK(x1, x2) / dx2[index2] for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.grad2(v1, v2, index, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

@jit
def _CovMatrix_Hess(X1: Array, X2: Array, kernel: BaseKernel, params: Array, index_1: int, index_2: int) -> Array:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 and X2 representing derivative values of the target function.

    Parameters
    ----------
    X1 : Array
        shape (N1, n_features)
    X2 : Array
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : Array
        kernel parameters
    index_1 : int
        one partial derivative of the kernel is taken w.r.t. X1[i,index1] 
    index_2 : int
        the other partial derivative of the kernel is taken w.r.t. X2[i,index2]

    Returns
    -------
    Array
        shape (N1, N2), [dK(x1, x2) / (dx1[index1]*dx2[index2]) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.jac(v1, v2, index_1, index_2, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)