from dataclasses import dataclass
from typing import Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.numpy import ndarray
from jaxopt import ScipyBoundedMinimize

from .. import covar, likelihood, predict
from ..kernels import BaseKernel
from ..logger import Logger
from .optim import optimize


@dataclass
class ExactGPR:
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
    optimize_noise: bool = False
    optimize_method: str = "L-BFGS-B"
    logger: Logger = None

    def train(self, X_data: Tuple[ndarray, ndarray], Y_data: ndarray) -> None:
        '''Fits a full gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Union[ndarray, list[ndarray]]
            shape either (n_samples, n_features) or [(n_samples_1, n_features), ..., (n_samples_N, n_features)]
            sum(n_samples_i) = n_samples. If given in form List[ndarray] the order must be 
            [function evals, derivative w.r.t. first feature, ..., derivative w.r.t. last feature]
        Y_data : ndarray
            shape (n_samples, ). labels corresponding to the elements in X_data
        '''
        self.X_split = X_data

        if self.optimize_noise:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                return likelihood.full_NLML(self.X_split, Y_data, self.kernel, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3)
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise)
        else:
            def optim_fun(kernel_params):
                return likelihood.full_NLML(self.X_split, Y_data, self.kernel, kernel_params, self.noise)  

            bounds = (1e-3, jnp.inf)  

            init_params = self.kernel_params  

        self.kernel_params = optimize(fun=optim_fun,
                                      params=init_params,
                                      bounds=bounds,
                                      method=self.optimize_method,
                                      callback=self.logger,
                                      jit_fun=True)

        self.covar_module = jit(covar.full_covariance_matrix)(self.X_split, self.noise, self.kernel, self.kernel_params)
        self.Y_data = Y_data

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
        return jit(predict.full_predict)(X, self.covar_module, self.Y_data, self.X_split, self.kernel, self.kernel_params)
    
@dataclass    
class SparseGPR:
    '''a sparse (PPA) Gaussian Process Regressor model.
    The full gaussian process is projected into a smaller subspace for computational efficiency
    
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
    X_ref : ndarray
        shape (n_referencepoints, n_features). Reference points onto which the gaussian process is projected.
    optimize_method : str, optional
        method to use in the optimizing process of the model parameters
    logger : Logger, optional
        If given must be a class with a __call__ method and a write method, by default None
        Logs the results of the optimization procedure.
    '''
    kernel: BaseKernel
    kernel_params: ndarray
    noise: Union[float, ndarray]
    X_ref: ndarray
    optimize_noise: bool = False
    optimize_ref: bool = False
    optimize_method:str = "L-BFGS-B"
    logger: Logger = None

    def train(self, X_data: Tuple[ndarray, ndarray], Y_data: ndarray) -> None:
        '''Fits a sparse (PPA) gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Union[ndarray
            shape either (n_samples, n_features) or [(n_samples_1, n_features), ..., (n_samples_N, n_features)]
            sum(n_samples_i) = n_samples. If given in form List[ndarray] the order must be 
            [function evals, derivative w.r.t. first feature, ..., derivative w.r.t. last feature]
        Y_data : ndarray
            shape (n_samples, ). labels corresponding to the elements in X_data
        '''
        self.X_split = X_data

        if self.optimize_noise and self.optimize_ref:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                X_ref = params[2]
                return likelihood.sparse_NLML(self.X_split, Y_data, self.kernel, X_ref, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3, jnp.ones_like(self.X_ref)*(-jnp.inf))
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf, jnp.ones_like(self.X_ref)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise, self.X_ref)
        elif self.optimize_ref:
            def optim_fun(params):
                kernel_params = params[0]
                X_ref = params[1]
                return likelihood.sparse_NLML(self.X_split, Y_data, self.kernel, X_ref, kernel_params, self.noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.X_ref)*(-jnp.inf))
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.X_ref)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.X_ref)
        elif self.optimize_noise:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                return likelihood.sparse_NLML(self.X_split, Y_data, self.kernel, self.X_ref, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3)
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise)
        else:
            def optim_fun(params):
                return likelihood.sparse_NLML(self.X_split, Y_data, self.kernel, self.X_ref, params, self.noise)

            bounds = (1e-3, jnp.inf)  

            init_params = self.kernel_params

        self.kernel_params = optimize(fun=optim_fun,
                                      params=init_params,
                                      bounds=bounds,
                                      method=self.optimize_method,
                                      callback=self.logger,
                                      jit_fun=True)

        self.covar_module = jit(covar.sparse_covariance_matrix)(self.X_split, Y_data, self.X_ref, self.noise, self.kernel, self.kernel_params)

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
        return jit(predict.sparse_predict)(X, self.covar_module, self.X_ref, self.kernel, self.kernel_params)