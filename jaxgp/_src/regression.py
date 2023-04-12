import jax.numpy as jnp
from jax import jit
from jaxopt import ScipyBoundedMinimize

from jax import Array
from typing import Union, Tuple

from .kernels import BaseKernel
from . import covar, predict, likelyhood

class ExactGPR:
    def __init__(self, kernel: BaseKernel, init_kernel_params: Array, noise: Union[float, Array], *, optimize_noise=False, logger=None) -> None:
        self.kernel = kernel
        self.kernel_params = jnp.array(init_kernel_params)
        self.noise = noise

        self.optimize_noise = optimize_noise
        self.logger = logger

    def train(self, X_data: Union[Array, list[Array]], Y_data: Array, *, data_split: Tuple = None) -> None:
        if data_split is None:
            self.X_split = X_data
        else:
            sum_splits = [jnp.sum(data_split[:i+1]) for i,_ in enumerate(data_split[1:])]
            self.X_split = jnp.split(X_data, sum_splits)

        solver = ScipyBoundedMinimize(fun=likelyhood.full_kernelNegativeLogLikelyhood, method="l-bfgs-b", callback=self.logger)
        self.kernel_params = solver.run(self.kernel_params, (1e-3,jnp.inf), self.X_split, Y_data, self.noise, self.kernel).params

        self.fit_matrix = covar.full_covariance_matrix(self.X_split, self.noise, self.kernel, self.kernel_params)
        self.fit_vector = Y_data

    def eval(self, X: Array) -> Tuple[Array, Array]:
        return predict.full_predict(X, self.fit_matrix, self.fit_vector, self.X_split, self.kernel, self.kernel_params)
    
class SparseGPR:
    def __init__(self, kernel: BaseKernel, init_kernel_params: Array, noise: Union[float, Array], X_ref: Array, *, optimize_noise=False, logger=None) -> None:
        self.kernel = kernel
        self.kernel_params = jnp.array(init_kernel_params)
        self.noise = noise

        self.X_ref = X_ref

        self.optimize_noise = optimize_noise
        self.logger = logger

    def train(self, X_data: Union[Array, list[Array]], Y_data: Array, *, data_split: Tuple = None) -> None:
        if data_split is None:
            self.X_split = X_data
        else:
            sum_splits = [jnp.sum(data_split[:i+1]) for i,_ in enumerate(data_split[1:])]
            self.X_split = jnp.split(X_data, sum_splits)

        solver = ScipyBoundedMinimize(fun=likelyhood.sparse_kernelNegativeLogLikelyhood, method="l-bfgs-b", callback=self.logger)
        self.kernel_params = solver.run(self.kernel_params, (1e-3,jnp.inf), self.X_split, Y_data, self.X_ref, self.noise, self.kernel).params

        self.fit_matrix, self.fit_vector = covar.sparse_covariance_matrix(self.X_split, Y_data, self.X_ref, self.noise, self.kernel, self.kernel_params)

    def eval(self, X: Array) -> Tuple[Array, Array]:
        return predict.sparse_predict(X, self.fit_matrix, self.fit_vector, self.X_ref, self.noise, self.kernel, self.kernel_params)