from typing import Tuple, Union

import jax.numpy as jnp
from jax import Array
from jaxopt import ScipyBoundedMinimize

from . import covar, likelyhood, predict
from .kernels import BaseKernel
from .logger import Logger


class ExactGPR:
    '''A full Gaussian Process regressor model
    '''
    def __init__(self, kernel: BaseKernel, init_kernel_params: Array, noise: Union[float, Array], *, optimize_method="l-bfgs-b", logger: Logger = None) -> None:
        '''
        Parameters
        ----------
        kernel : derived class from BaseKernel
            Kernel that describes the covariance between input points.
        init_kernel_params : Array
            initial kernel parameters for the optimization
        noise : Union[float, Array]
            noise present in the input labels
            either scalar or Array of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
            Else each value is added to the corresponding diagonal block coming from X_split
        optimize_method : str, optional
            method to use in the optimizing process of the model parameters
        logger : Logger, optional
            If given must be a class with a __call__ method and a write method, by default None
            Logs the results of the optimization procedure.
        '''
        self.kernel = kernel
        self.kernel_params = jnp.array(init_kernel_params)
        self.noise = noise

        self.optimize_method = optimize_method
        self.logger = logger

    def train(self, X_data: Tuple[Array, Array], Y_data: Array) -> None:
        '''Fits a full gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Union[Array, list[Array]]
            shape either (n_samples, n_features) or [(n_samples_1, n_features), ..., (n_samples_N, n_features)]
            sum(n_samples_i) = n_samples. If given in form List[Array] the order must be 
            [function evals, derivative w.r.t. first feature, ..., derivative w.r.t. last feature]
        Y_data : Array
            shape (n_samples, ). labels corresponding to the elements in X_data
        data_split : Tuple, optional
            shape (1 + n_features, ) if X_data is Array, None if X_data is List[Array]
            describes the how many of each type of evaluation in X_data are present.
        '''
        self.X_split = X_data

        solver = ScipyBoundedMinimize(fun=likelyhood.full_kernelNegativeLogLikelyhood, method=self.optimize_method, callback=self.logger)
        result = solver.run(self.kernel_params, (1e-3,jnp.inf), self.X_split, Y_data, self.noise, self.kernel)
        print(result)
        self.kernel_params = result.params
        if self.logger is not None:
            self.logger.write()

        self.fit_matrix = covar.full_covariance_matrix(self.X_split, self.noise, self.kernel, self.kernel_params)
        self.fit_vector = Y_data

    def eval(self, X: Array) -> Tuple[Array, Array]:
        '''evaluates the posterior mean and std for each point in X

        Parameters
        ----------
        X : Array
            shape (N, n_features). Set of points for which to calculate the posterior means and stds

        Returns
        -------
        Tuple[Array, Array]
            Posterior means and stds
        '''
        return predict.full_predict(X, self.fit_matrix, self.fit_vector, self.X_split, self.kernel, self.kernel_params)
    
class SparseGPR:
    '''a sparse (PPA) Gaussian Process Regressor model.
    The full gaussian process is projected into a smaller subspace for computational efficiency
    '''
    def __init__(self, kernel: BaseKernel, init_kernel_params: Array, noise: Union[float, Array], X_ref: Array, *, optimize_method="l-bfgs-b", logger=None) -> None:
        '''
        Parameters
        ----------
        kernel : derived class from BaseKernel
            Kernel that describes the covariance between input points.
        init_kernel_params : Array
            initial kernel parameters for the optimization
        noise : Union[float, Array]
            noise present in the input labels
            either scalar or Array of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
            Else each value is added to the corresponding diagonal block coming from X_split
        X_ref : Array
            shape (n_referencepoints, n_features). Reference points onto which the gaussian process is projected.
        optimize_method : str, optional
            method to use in the optimizing process of the model parameters
        logger : Logger, optional
            If given must be a class with a __call__ method and a write method, by default None
            Logs the results of the optimization procedure.
        '''
        self.kernel = kernel
        self.kernel_params = jnp.array(init_kernel_params)
        self.noise = noise

        self.X_ref = X_ref

        self.optimize_method = optimize_method
        self.logger = logger

    def train(self, X_data: Tuple[Array, Array], Y_data: Array) -> None:
        '''Fits a sparse (PPA) gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Union[Array, list[Array]]
            shape either (n_samples, n_features) or [(n_samples_1, n_features), ..., (n_samples_N, n_features)]
            sum(n_samples_i) = n_samples. If given in form List[Array] the order must be 
            [function evals, derivative w.r.t. first feature, ..., derivative w.r.t. last feature]
        Y_data : Array
            shape (n_samples, ). labels corresponding to the elements in X_data
        data_split : Tuple, optional
            shape (1 + n_features, ) if X_data is Array, None if X_data is List[Array]
            describes the how many of each type of evaluation in X_data are present.
        '''
        self.X_split = X_data

        solver = ScipyBoundedMinimize(fun=likelyhood.sparse_kernelNegativeLogLikelyhood, method=self.optimize_method, callback=self.logger)
        result = solver.run(self.kernel_params, (1e-3,jnp.inf), self.X_split, Y_data, self.X_ref, self.noise, self.kernel)
        print(result)
        self.kernel_params = result.params
        if self.logger is not None:
            self.logger.write()
        self.fit_matrix, self.fit_vector = covar.sparse_covariance_matrix(self.X_split, Y_data, self.X_ref, self.noise, self.kernel, self.kernel_params)

    def eval(self, X: Array) -> Tuple[Array, Array]:
        '''evaluates the posterior mean and std for each point in X

        Parameters
        ----------
        X : Array
            shape (N, n_features). Set of points for which to calculate the posterior means and stds

        Returns
        -------
        Tuple[Array, Array]
            Posterior means and stds
        '''
        return predict.sparse_predict(X, self.fit_matrix, self.fit_vector, self.X_ref, self.noise, self.kernel, self.kernel_params)