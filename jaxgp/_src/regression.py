import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit, vmap
from functools import partial
from jaxopt import ScipyBoundedMinimize

from jax import Array
from typing import Union, Tuple

from .kernels import RBF
from . import covar, predict, likelyhood
from .logger import Logger

class BaseGPR:
    def __init__(self, kernel=RBF(), data_split=(0, 0), init_kernel_params=(1.0, 1.0), noise=1e-4, *, optimize_noise=False) -> None:
        '''
            supported kernels are found in gaussion_process.kernels

            data_split : Iterable       ... first element:      number of (noisy) function evaluations
                                            further elements:   number of (noisy) derivative evaluations
            kernel_params : Iterable    ... initial parameters for the MLE parameter optimization
            noise_var : float           ... assumed noise in the evaluation of data
        '''
        self.kernel = kernel
        self.data_split = jnp.array(data_split)

        # defines parameters of the regressor and sets flag for optimizing the noise parameter as well
        self.optimize_noise = optimize_noise
        self.kernel_params = init_kernel_params
        self.noise = noise
        self.params = jnp.array((noise,) + init_kernel_params)

        # initialized variables to save from the fitting step
        self.fit_vector = None
        self.fit_matrix = None
        self.further_args = []

        self.logger = Logger()

        # TODO:
        # add functionality to the prior distributions

    def train(self, X_data: Array, Y_data: Array) -> None:
        '''
            Fits the GPR Model to the given data

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)

            Thin wrapper with all side effects around the pure functions 
            self.optimize and self.forward that do the actual work.
        '''
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        self.further_args.append(jnp.split(X_data, sum_splits))

        self.Y_data = Y_data

        self.optimize(Y_data, *self.further_args)
        self.fit_matrix, self.fit_vector = self.forward(self.params, Y_data, *self.further_args)

    def forward(self, params: Array, Y_data: Array, *further_args) -> Tuple[Array, Array]:
        raise NotImplementedError("Forward method not yet implemented!")
    
    def eval(self, X: Array) -> Union[Tuple[Array, Array], Array]:
        '''
            Predicts the posterior mean (and std if return_std=True) at each point in X
            X.shape = (N, n_features)
            
            Thin wrapper with all side effects around the pure function 
            self.eval that does the actual work.
        '''
        if self.fit_matrix is None or self.fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        
        return self._meanstd(self.params, X, self.fit_matrix, self.fit_vector, *self.further_args)
    
    def _meanstd(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, *further_args) -> Tuple[Array, Array]: raise NotImplementedError
    def _negativeLogLikelyhood(self, params: Array, Y_data: Array, *further_args) -> float: raise NotImplementedError
    def _kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, *further_args) -> float: raise NotImplementedError

    def optimize(self, Y_data: Array, *further_args) -> None:
        '''
            Maximizes the log likelyhood function either for 
                the kernel parameters or
                the noise and the kernel parameters
        '''
        if self.optimize_noise:
            solver = ScipyBoundedMinimize(fun=self._negativeLogLikelyhood, method="l-bfgs-b", callback=self.logger)
            result = solver.run(self.params, (1e-3,jnp.inf), Y_data, *further_args)
            
            loss = lambda x: self._negativeLogLikelyhood(x, self.noise, self.Y_data, *self.further_args)*20000.0
            self.logger.write(loss)

            print(result)

            self.params = result.params
        else:
            solver = ScipyBoundedMinimize(fun=self._kernelNegativeLogLikelyhood, method="l-bfgs-b", callback=self.logger)
            result = solver.run(self.kernel_params, (1e-3,jnp.inf), self.noise, Y_data, *further_args)
            
            loss = lambda x: self._kernelNegativeLogLikelyhood(x, self.noise, self.Y_data, *self.further_args)*20000.0
            self.logger.write(loss)

            print(result)

            self.params = jnp.array((self.noise, ) + result.params)

# class ExactGPR:
#     def __init__(self, kernel, init_kernel_params, noise, *, optimize_noise) -> None:
#         self.kernel = kernel
#         self.kernel_params = init_kernel_params
#         self.noise = noise

#         self.optimize_noise = optimize_noise

#     def logger(self, output):
#         print(output)

#     def train(self, X_data, Y_data, *, data_split=None):
#         if data_split is None:
#             X_split = X_data
#         else:
#             sum_splits = [jnp.sum(data_split[:i+1]) for i,_ in enumerate(data_split[1:])]
#             X_split = jnp.split(X_data, sum_splits)

#         solver = ScipyBoundedMinimize(fun=likelyhood.full_kernelNegativeLogLikelyhood, method="l-bfgs-b")
#         self.kernel_params = solver.run(self.kernel_params, (1e-3,jnp.inf), X_split, Y_data, self.noise, self.kernel).params

#         self.fit_matrix = covar.full_covariance_matrix(X_split, self.noise, self.kernel, self.kernel_params)
#         self.fit_vector = Y_data

class ExactGPR(BaseGPR):
    def __init__(self, kernel=RBF(), data_split=(0, 0), kernel_params=(1.0,), noise=1e-4, *, optimize_noise=False) -> None:
        super().__init__(kernel, data_split, kernel_params, noise, optimize_noise=optimize_noise)

        self.further_args = []

    def forward(self, params: Array, Y_data: Array, X_split: list[Array]) -> Tuple[Array, Array]:
        fitmatrix = jit(covar.full_covariance_matrix)(X_split, params[0], self.kernel, params[1:])
        return fitmatrix, Y_data
    
    def _meanstd(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, X_split: Array) -> Tuple[Array, Array]:
        return jit(predict.full_predict)(X, fitmatrix, fitvector, X_split, self.kernel, params[1:])
    
    def _kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, X_split: list[Array]) -> float:
        return jit(likelyhood.full_kernelNegativeLogLikelyhood)(kernel_params, X_split, Y_data, noise, self.kernel)

class SparseGPR(BaseGPR):
    def __init__(self, kernel=RBF(), data_split=(0,), X_ref=None, kernel_params=(1.0,), noise= 1e-4, *, optimize_noise=False) -> None:
        '''
            Sparsifies the full GP regressor by the Projected Process Approximation.
        '''
        super().__init__(kernel, data_split, kernel_params, noise, optimize_noise=optimize_noise)

        self.further_args = [X_ref, ]

    def forward(self, params: Array, Y_data: Array, X_ref: Array, X_split: list[Array]) -> Tuple[Array, Array]:
        return jit(covar.sparse_covariance_matrix)(X_split, Y_data, X_ref, params[0], self.kernel, params[1:])
    
    # @partial(jit, static_argnums=(0,))
    def _meanstd(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, X_ref: Array, X_split: list[Array]) -> Tuple[Array, Array]:
        '''
            calculates the posterior mean and std for all points in X
        '''
        return jit(predict.sparse_predict)(X, fitmatrix, fitvector, X_ref, params[0], self.kernel, params[1:])
    
    def _kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, X_ref: Array, X_split: list[Array]) -> float:
        return jit(likelyhood.sparse_kernelNegativeLogLikelyhood)(kernel_params, X_split, Y_data, X_ref, noise, self.kernel)