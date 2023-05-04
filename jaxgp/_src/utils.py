from functools import partial

from jax import jit, vmap
from jax.numpy import ndarray
from jax.scipy.linalg import solve

from .kernels import BaseKernel


@jit
@partial(vmap, in_axes=(None, 0))
def _build_xT_Ainv_x(A: ndarray, X: ndarray) -> ndarray:
    ''' Calculates X.T @ A_inv @ X for each of the M arrays in X

    Parameters
    ----------
    A : ndarray
        positive definite matrix of shape (N, N)
    X : ndarray
        M arrays of shape (N, )

    Returns
    -------
    ndarray
        shape (M, ), [x.T @ A_inv @ x for x in X]
    '''
    return X.T@solve(A,X,assume_a="pos")

@jit
def _CovVector_Id(X: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Calculates the covariance of each point in X with itself

    Parameters
    ----------
    X : ndarray
        array of shape (N, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N, ), [K(x, x) for x in X]
    '''
    func = lambda v: kernel.eval(v, v, params)
    func = vmap(func, in_axes=(0))
    return func(X)

@jit
def _CovMatrix_Kernel(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on inputs representing values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1, N2), [K(x1, x2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.eval(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

@jit
def _CovMatrix_Grad(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 representing values of the target function and X2
    representing derivative values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters
    index : int
        derivative of the kernel is calculted w.r.t. to X2[index]

    Returns
    -------
    ndarray
        shape (N1, N2), [dK(x1, x2) / dx2[index2] for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.grad2(v1, v2, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)

@jit
def _CovMatrix_Hess(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 and X2 representing derivative values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of BaseKernel
        Kernel that describes the covariance between input points.
    params : ndarray
        kernel parameters
    index_1 : int
        one partial derivative of the kernel is taken w.r.t. X1[i,index1] 
    index_2 : int
        the other partial derivative of the kernel is taken w.r.t. X2[i,index2]

    Returns
    -------
    ndarray
        shape (N1, N2), [dK(x1, x2) / (dx1[index1]*dx2[index2]) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.jac(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)