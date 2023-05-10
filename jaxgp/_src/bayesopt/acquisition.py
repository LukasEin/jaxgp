from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit, random
from jax.numpy import ndarray
from jaxopt import ScipyBoundedMinimize

from ..kernels import BaseKernel
from ..predict import full_predict

# def upper_confidence_bound(cov_matrix, Y_data, X_split, kernel, params, grid, eval_function):
        
#     mean, std = full_predict(grid, cov_matrix, Y_data, X_split, kernel, params)
#     maximizer = mean + std

#     maxarg = jnp.argmax(maximizer)

#     X_next = grid[maxarg]
#     Y_next, isgrad = eval_function(X_next)

#     return X_next, Y_next, isgrad

# def upper_confidence_bound(cov_matrix, Y_data, X_split, kernel, params, bounds, eval_function):
#     def minim_func(x):
#         mean, std = full_predict(x, cov_matrix, Y_data, X_split, kernel, params)
#         return -std[0]
    
#     key = random.PRNGKey(0)
#     init_point = random.uniform(key, shape=bounds[0].shape, minval=bounds[0], maxval=bounds[1]).reshape(1,-1)
    
#     solver = ScipyBoundedMinimize(fun=minim_func, method="L-BFGS-B")
#     result = solver.run(init_point, bounds)

#     X_next = result.params
#     Y_next, isgrad = eval_function(X_next)

#     return X_next, Y_next.reshape(-1), isgrad

def upper_confidence_bound(cov_matrix: ndarray, Y_data: ndarray, X_split: Tuple[ndarray, ndarray], kernel: BaseKernel, kernel_params: ndarray, 
                           index: int, rand: int, bounds: ndarray, eval_function: Callable, explorparam: float, grid: ndarray) -> Tuple[Tuple[ndarray, ndarray], ndarray, bool]:
    '''_summary_

    Parameters
    ----------
    cov_matrix : ndarray
        _description_
    Y_data : ndarray
        _description_
    X_split : Tuple[ndarray, ndarray]
        _description_
    kernel : BaseKernel
        _description_
    kernel_params : ndarray
        _description_
    index : int
        _description_
    rand : int
        _description_
    bounds : ndarray
        _description_
    eval_function : Callable
        _description_
    explorparam : float
        _description_
    grid : ndarray
        _description_

    Returns
    -------
    Tuple[Tuple[ndarray, ndarray], ndarray, bool]
        _description_
    '''
    if index%rand == 0:
        key = random.PRNGKey(index*rand)
        X_next = random.uniform(key, shape=bounds[0].shape, minval=bounds[0], maxval=bounds[1]).reshape(1,-1)
        Y_next, isgrad = eval_function(X_next)
    else:
        mean, std = jit(full_predict)(grid.reshape(-1,1), cov_matrix, Y_data, X_split, kernel, kernel_params)

        next_arg = jnp.argmax((mean + explorparam*std).reshape(-1))
        X_next = grid[next_arg].reshape(1,-1)
        Y_next, isgrad = eval_function(X_next)

    return X_next, Y_next.reshape(-1), isgrad