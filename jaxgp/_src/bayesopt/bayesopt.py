from dataclasses import dataclass
from typing import Callable, Tuple

from jax import jit
import jax.numpy as jnp
from jax.numpy import ndarray

from ..kernels import BaseKernel
from ..utils import for_loop
from ..regression.regression import ExactGPR


def _step_full_grad(X_split, Y_data, gpr, acqui_fun, eval_fun):
    # Find new best point
    X_next = acqui_fun(gpr)
    Y_next = eval_fun(X_next)

    # update data
    X_next = (X_split[0], jnp.vstack((X_split[1], X_next)))
    Y_next = (Y_data[0], jnp.vstack((Y_data[1], Y_next)))

    return X_next, Y_next

def _step_full_fun(X_split, Y_data, gpr, acqui_fun, eval_fun):
    # Find new best point
    X_next = acqui_fun(gpr)
    Y_next = eval_fun(X_next)

    # update data
    X_next = (jnp.vstack((X_split[0], X_next)), X_split[1])
    Y_next = (jnp.vstack((Y_data[0], Y_next)), Y_data[1])

    return X_next, Y_next

@dataclass   
class ExactBayesOpt:
    X_split: Tuple[ndarray, ndarray]
    Y_train: Tuple[ndarray, ndarray]
    kernel: BaseKernel
    acquisition_func: Callable
    eval_func: Callable
    grad: bool = True
    optimize_method: str = 0
    logger: Callable = None

    def __post_init__(self) -> None:
        self.gpr = ExactGPR(self.kernel, optimize_method=self.optimize_method)

    def __call__(self, num_iters: int) -> None:
        def acqui_fun(gpr):
            return self.acquisition_func(gpr, self.logger)

        if self.grad:
            def _body_fun(X, Y, gpr):
                return _step_full_grad(X, Y, gpr, acqui_fun, self.eval_func)
        else:
            def _body_fun(X, Y, gpr):
                return _step_full_fun(X, Y, gpr, acqui_fun, self.eval_func)
            
        body_fun = jit(_body_fun)

        X, Y = self.X_split, self.Y_train

        for _ in range(num_iters):
            self.gpr.reset_params()
            self.gpr.train(X, jnp.vstack(Y).reshape(-1))
            X, Y = body_fun(X, Y, self.gpr)

        self.X_split, self.Y_train = X, Y
        