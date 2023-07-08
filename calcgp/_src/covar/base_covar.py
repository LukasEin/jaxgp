import jax.numpy as jnp
from jax import vmap
from jax.numpy import ndarray

from ..kernels import Kernel


def CovDiagF(X: ndarray, kernel: Kernel, kernel_params: ndarray) -> ndarray:
    '''Calculates the covariance of each point in X with itself

    Parameters
    ----------
    X : ndarray
        array of shape (N, n_features)
    kernel : derived class of Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N, ), [K(x, x) for x in X]
    '''
    func = lambda v: kernel.eval(v, v, kernel_params)
    func = vmap(func, in_axes=(0))
    return func(X)



def CovDiagG(X: ndarray, kernel: Kernel, kernel_params: ndarray) -> ndarray:
    '''Calculates the covariance of each point in X with itself

    Parameters
    ----------
    X : ndarray
        array of shape (N, n_features)
    kernel : derived class of Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N, ), [K(x, x) for x in X]
    '''
    func = lambda v: jnp.diag(kernel.jac(v, v, kernel_params))
    func = vmap(func, in_axes=(0))
    return jnp.ravel(func(X))



def CovMatrixFF(X1: ndarray, X2: ndarray, kernel: Kernel, kernel_params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on inputs representing values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1, N2), [K(x1, x2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.eval(v1, v2, kernel_params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return func(X1, X2)



def CovMatrixFG(X1: ndarray, X2: ndarray, kernel: Kernel, kernel_params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 representing values of the target function and X2
    representing gradient values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1, N2 * n_features), [dK(x1, x2) / dx2 for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.grad2(v1, v2, kernel_params) 
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return vmap(jnp.ravel, in_axes=0)(func(X1, X2))



def CovMatrixGG(X1: ndarray, X2: ndarray, kernel: Kernel, kernel_params: ndarray) -> ndarray:
    '''Builds the covariance matrix between the elements of X1 and X2
    based on X1 and X2 representing gradient values of the target function.

    Parameters
    ----------
    X1 : ndarray
        shape (N1, n_features)
    X2 : ndarray
        shape (N2, n_features)
    kernel : derived class of Kernel
        Kernel that describes the covariance between input points.
    kernel_params : ndarray
        kernel parameters

    Returns
    -------
    ndarray
        shape (N1 * n_features, N2 * n_features), [dK(x1, x2) / (dx1*dx2) for (x1, x2) in (X1, X2)]
    '''
    func = lambda v1, v2: kernel.jac(v1, v2, kernel_params)
    func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
    return jnp.hstack(jnp.hstack((*func(X1, X2),)))