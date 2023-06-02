import jax.scipy as jsp
from jax import vmap
from jax.numpy import hstack, ndarray, ravel

from .kernels import BaseKernel


def _matmul_diag(diagonal: ndarray, rhs: ndarray) -> ndarray:
    '''Faster matrix multiplication for a diagonal matrix. 

    Parameters
    ----------
    diagonal : ndarray
        shape (N,). A diagonal matrix represented by a 1d vector
    rhs : ndarray
        shape (N, M). A generic matrix to be multiplied with a diagonal matrix from the left

    Returns
    -------
    ndarray
        shape (N, M). Product matrix
    '''
    return diagonal*rhs

matmul_diag = vmap(_matmul_diag, in_axes=(0,0))

def _inner_map(lower_triangular: ndarray, rhs: ndarray) -> ndarray:
    '''Use to calculate the inner product x.T @ A^-1 @ x were A is positive definite.
    A must be given in its lower Cholesky decomposition L. 
    The result is mapped over the second axis of x.

    Parameters
    ----------
    lower_triangular : ndarray
        shape (N, N). Lower triagular Cholesky decomposition of a pos def matrix.
    rhs : ndarray
        shape (N, M). Set of vectors over which the inner product is mapped

    Returns
    -------
    ndarray
        shape (M,). 
    '''
    sol = jsp.linalg.solve_triangular(lower_triangular.T,rhs, lower=True)
    return sol.T@sol

inner_map = vmap(_inner_map, in_axes=(None, 0))

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

def CovMatrixFF(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
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

def CovMatrixFD(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 representing values of the target function and X2
    representing gradient values of the target function.

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
        shape (N1, N2 * n_features), [dK(x1, x2) / dx2 for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.grad2(v1, v2, params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return vmap(ravel, in_axes=0)(func(X1, X2))

def CovMatrixDD(X1: ndarray, X2: ndarray, kernel: BaseKernel, params: ndarray) -> ndarray:
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

    Returns
    -------
    ndarray
        shape (N1 * n_features, N2 * n_features), [dK(x1, x2) / (dx1*dx2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.jac(v1, v2, params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return hstack(hstack((*func(X1, X2),)))