from typing import Callable, Any

import jax.scipy as jsp
from jax import jit, vmap
from jax.numpy import ndarray


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

def _for_loop(lb: int, ub: int, init_val: Any, body_fun: Callable) -> Any:
    val = init_val

    for i in range(lb, ub):
        val = body_fun(i, val)

    return val

for_loop = jit(_for_loop, static_argnums=(0,1,3))