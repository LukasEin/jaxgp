from dataclasses import dataclass
from typing import Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.numpy import ndarray
from jaxopt import ScipyBoundedMinimize

from .. import covar, likelihood, predict
from ..kernels import BaseKernel
from ..logger import Logger
from .predict_nograd import *
from .covar_nograd import *
from .likelihood_nograd import *
    
@dataclass
class GPRnograd:
    '''A full Gaussian Process regressor model

    Parameters
    ----------
    kernel : derived class from BaseKernel
        Kernel that describes the covariance between input points.
    init_kernel_params : ndarray
        initial kernel parameters for the optimization
    noise : Union[float, ndarray]
        noise present in the input labels
        either scalar or ndarray of shape (len(X_split),). If scalar, the same value is added along the diagonal. 
        Else each value is added to the corresponding diagonal block coming from X_split
    optimize_method : str, optional
        method to use in the optimizing process of the model parameters
    logger : Logger, optional
        If given must be a class with a __call__ method and a write method, by default None
        Logs the results of the optimization procedure.
    '''
    kernel: BaseKernel
    kernel_params: ndarray
    noise: Union[float, ndarray]
    X_ref: ndarray = None
    is_sparse: bool = False
    optimize_method:str = "L-BFGS-B"
    logger: Logger = None

    def train(self, X_data: ndarray, Y_data: ndarray) -> None:
        '''Fits a full gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Union[ndarray, list[ndarray]]
            shape either (n_samples, n_features) or [(n_samples_1, n_features), ..., (n_samples_N, n_features)]
            sum(n_samples_i) = n_samples. If given in form List[ndarray] the order must be 
            [function evals, derivative w.r.t. first feature, ..., derivative w.r.t. last feature]
        Y_data : ndarray
            shape (n_samples, ). labels corresponding to the elements in X_data
        data_split : Tuple, optional
            shape (1 + n_features, ) if X_data is ndarray, None if X_data is List[ndarray]
            describes the how many of each type of evaluation in X_data are present.
        '''
        self.X_data = X_data

        if self.is_sparse:
            solver = ScipyBoundedMinimize(fun=jit(sparse_kernelNegativeLogLikelyhood_nograd), method=self.optimize_method, callback=self.logger)
            result = solver.run(self.kernel_params, (1e-3,jnp.inf), self.X_data, Y_data, self.X_ref, self.noise, self.kernel)
        else:
            solver = ScipyBoundedMinimize(fun=jit(full_kernelNegativeLogLikelyhood_nograd), method=self.optimize_method, callback=self.logger)
            result = solver.run(self.kernel_params, (1e-3,jnp.inf), self.X_data, Y_data, self.noise, self.kernel)
        
        print(result)
        self.kernel_params = result.params
        if self.logger is not None:
            self.logger.write()

        if self.is_sparse:
            self.fit_matrix, self.fit_vector = jit(sparse_covariance_matrix_nograd)(self.X_data, Y_data, self.X_ref, self.noise, self.kernel, self.kernel_params)
        else:
            self.fit_matrix = jit(full_covariance_matrix_nograd)(self.X_data, self.noise, self.kernel, self.kernel_params)
            self.fit_vector = Y_data

    def eval(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        '''evaluates the posterior mean and std for each point in X

        Parameters
        ----------
        X : ndarray
            shape (N, n_features). Set of points for which to calculate the posterior means and stds

        Returns
        -------
        Tuple[ndarray, ndarray]
            Posterior means and stds
        '''
        if self.is_sparse:
            return jit(sparse_predict_nograd)(X, self.fit_matrix, self.fit_vector, self.X_ref, self.noise, self.kernel, self.kernel_params)
        else:
            return jit(full_predict_nograd)(X, self.fit_matrix, self.fit_vector, self.X_data, self.kernel, self.kernel_params)