from dataclasses import dataclass
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax.numpy import ndarray

from ..kernels import BaseKernel
from ..covar import FullCovar, SparseCovar
from ..utils import *


def _step_full_grad(covar_module: FullCovar, acqui_fun, X_split, Y_data, kernel, kernel_params):
    # Find new best point
    X_next, Y_next = acqui_fun()

    # update covariance matrix
    cov_self = CovMatrixDD(X_next, X_next, kernel, kernel_params)

    cov_grad = CovMatrixDD(X_split[1], X_next, kernel, kernel_params)
    cov_fun = CovMatrixFD(X_split[0], X_next, kernel, kernel_params)
    cov_other = jnp.vstack((cov_fun, cov_grad))

    cov = jnp.vstack((jnp.hstack((covar_module.k_nn, cov_other)),
                      jnp.hstack((cov_other.T, cov_self))))

    # update data
    X_next = (X_split[0], jnp.vstack((X_split[1], X_next)))
    Y_next = (Y_data[0], jnp.vstack((Y_data[1], Y_next)))

    return FullCovar(cov), X_next, Y_next

def _step_full_fun(covar_module: FullCovar, acqui_fun, X_split, Y_data, kernel, kernel_params):
    # Find new best point
    X_next, Y_next = acqui_fun()

    # update covariance matrix
    cov_self = CovMatrixFF(X_next, X_next, kernel, kernel_params)

    cov_grad = CovMatrixFD(X_next, X_split[1], kernel, kernel_params)
    cov_fun = CovMatrixFF(X_next, X_split[0], kernel, kernel_params)
    cov_other = jnp.hstack((cov_fun, cov_grad))

    cov = jnp.vstack((jnp.hstack((cov_self, cov_other)),
                      jnp.hstack((cov_other.T, covar_module.k_nn))))

    # update data
    X_next = (jnp.vstack((X_split[0], X_next)), X_split[1])
    Y_next = (jnp.vstack((Y_data[0], Y_next)), Y_data[1])

    return FullCovar(cov), X_next, Y_next

def _for_loop(lb, ub, init_val, body_fun):
    val = init_val

    for i in range(lb, ub):
        val = body_fun(i, val)

    return val

for_loop = jax.jit(_for_loop, static_argnums=(0,1,3))

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
        