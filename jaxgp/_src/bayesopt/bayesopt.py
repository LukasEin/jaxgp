from dataclasses import dataclass
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from jaxopt import ScipyBoundedMinimize

from ..covar import full_covariance_matrix, full_covariance_matrix_nograd
from ..kernels import BaseKernel
from ..likelihood import full_kernelNegativeLogLikelyhood


def _bayesoptstep(X_split: Tuple[ndarray, ndarray], Y_data: Tuple[ndarray, ndarray], init_params: ndarray, kernel: BaseKernel, 
                  noise: Union[float, ndarray], optimize_method: str, acquisition_func: Callable, *args) -> ndarray:
    '''_summary_

    Parameters
    ----------
    X_split : Tuple(ndarray, ndarray)
        _description_
    Y_data : Tuple(ndarray, ndarray)
        _description_
    init_params : ndarray
        _description_
    kernel : BaseKernel
        _description_
    noise : Union[float, ndarray]
        _description_
    optimize_method : str
        _description_
    acquisition_func : Callable
        A function that takes the model as input and return the "best" next point to add to the model, i.e.
        model -> (X_next, Y_next / grad(Y_next), isgrad)
    args : Any
        Additional arguments for acquisition_func

    Returns
    -------
    ndarray
        _description_
    '''
    # solver = ScipyBoundedMinimize(fun=full_kernelNegativeLogLikelyhood, method=optimize_method)
    # result = solver.run(init_params, (1e-3,jnp.inf), X_split, jnp.vstack(Y_data), noise, kernel)

    # cov_matrix = full_covariance_matrix_nograd(X_split[1], noise, kernel, result.params)

    X_next, Y_next, isgrad = acquisition_func(X_split, Y_data, init_params, kernel, noise, optimize_method, *args)

    if isgrad:
        X_next = (X_split[0], jnp.vstack((X_split[1], X_next)))
        Y_next = (Y_data[0], jnp.vstack((Y_data[1], Y_next)))
    else:
        X_next = (jnp.vstack((X_split[0], X_next)), X_split[1])
        Y_next = (jnp.vstack((Y_data[0], Y_next)), Y_data[1])
        
    return (X_next, Y_next)

@dataclass   
class BayesOpt:
    X_split: Tuple[ndarray, ndarray]
    Y_train: ndarray
    kernel: BaseKernel
    init_kernel_params: ndarray
    noise: Union[float, ndarray]
    acquisition_func: Callable
    acqu_extra_args: Tuple
    optimize_method: str = "L-BFGS-B"
    logger: Callable = None

    # def __post_init__(self):
    #     self.result = (self.X_split, self.Y_train)

    def run(self, num_iters: int) -> None:
        X, Y = self.X_split, self.Y_train

        for i in range(num_iters):
            X, Y = _bayesoptstep(X, Y, self.init_kernel_params, self.kernel, self.noise, self.optimize_method, self.acquisition_func, i, *self.acqu_extra_args)

        self.X_split, self.Y_train = X, Y
        # def for_body(i, input):
        #     return _bayesoptstep(*input, self.init_kernel_params, self.kernel, self.noise, self.optimize_method, self.acquisition_func, *self.acqu_extra_args)
        
        # self.result = jax.lax.fori_loop(0, num_iters, for_body, self.result)
        