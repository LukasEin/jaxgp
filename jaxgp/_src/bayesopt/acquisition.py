from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
from jax.numpy import ndarray

from ..predict import full_predict
from ..regression.regression import ExactGPR
from ..regression.optim import optimize

from dataclasses import dataclass

@dataclass
class UpperConfidenceBound:
    bounds: Tuple
    num: float
    explore: float

    def __call__(self, gpr: ExactGPR, logger) -> ndarray:
        grid = jnp.linspace(self.bounds[0], self.bounds[1], self.num)
        mean, std = full_predict(grid, gpr.covar_module, gpr.X_split, gpr.kernel, gpr.kernel_params)

        ucb = mean + self.explore*std

        next_arg = jnp.argmax((ucb).reshape(-1))
        x_next = grid[next_arg].reshape(1,-1)

        return x_next
    
@dataclass
class MaximumVariance:
    bounds: Tuple
    num: float

    def __call__(self, gpr: ExactGPR, logger) -> ndarray:
        grid = jnp.linspace(self.bounds[0], self.bounds[1], self.num)
        _, std = full_predict(grid, gpr.covar_module, gpr.X_split, gpr.kernel, gpr.kernel_params)


        next_arg = jnp.argmax((std).reshape(-1))
        x_next = grid[next_arg].reshape(1,-1)

        return x_next

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

# def maximum_confidence_grad(X_split: Tuple[ndarray, ndarray], Y_data: Tuple[ndarray, ndarray], init_params: ndarray, kernel: BaseKernel, 
#                   noise: Union[float, ndarray], optimize_method: str, acquisition_func: Callable, grid: ndarray, eval_function) -> Tuple[ndarray, ndarray, bool]:
#     '''_summary_

#     Parameters
#     ----------
#     cov_matrix : ndarray
#         _description_
#     Y_data : Tuple[ndarray, ndarray]
#         _description_
#     X_split : Tuple[ndarray, ndarray]
#         _description_
#     kernel : BaseKernel
#         _description_
#     kernel_params : ndarray
#         _description_
#     eval_function : Callable
#         _description_
#     grid : ndarray
#         _description_

#     Returns
#     -------
#     Tuple[Tuple[ndarray, ndarray], ndarray, bool]
#         _description_
#     '''
#     solver = ScipyBoundedMinimize(fun=likelihood.full_kernelNegativeLogLikelyhood, method=optimize_method)
#     result = solver.run(init_params, (1e-3,jnp.inf), X_split, jnp.vstack(Y_data), noise, kernel)
    
#     cov_matrix = covar.full_covariance_matrix_nograd(X_split[1], noise, kernel, result.params)
    
#     predict_grad = lambda Y: full_predict_nograd(grid.reshape(-1,1), cov_matrix, Y, X_split[1], kernel, result.params)
#     _, std = jit(vmap(predict_grad, in_axes=(1,)))(Y_data[1])

#     next_arg = jnp.argmax((std).reshape(-1))
#     X_next = grid[next_arg].reshape(1,-1)
#     Y_next, isgrad = eval_function(X_next)

#     return X_next, Y_next.reshape(-1), isgrad