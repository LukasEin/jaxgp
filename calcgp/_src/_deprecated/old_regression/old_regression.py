from dataclasses import dataclass
from typing import Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.numpy import ndarray
from jax.tree_util import register_pytree_node_class

from .. import old_covar, old_likelihood, old_predict
from ..kernels import Kernel
from ..logger import Logger
from .old_optim import optimize, Optimizer

@register_pytree_node_class
@dataclass
class ExactGPRGrad:
    kernel: Kernel
    kernel_params: Union[float, ndarray] = jnp.log(2)
    noise: float = 1e-4
    optimize_noise: bool = False
    optimize_method: Optimizer = Optimizer.SLSQP
    logger: Logger = None

    def __post_init__(self) -> None:
        if jnp.isscalar(self.kernel_params):
            self.kernel_params = jnp.ones(self.kernel.num_params)*self.kernel_params

        self.X_split = None
        self.covar_module = None

    def train(self, X_data: ndarray, Y_data: ndarray) -> None:
        '''Fits a full gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Tuple[ndarray, ndarray]
            Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
        Y_data : ndarray
            shape (n_function_evals + n_gradient_evals, ). Input labels representing noisy function evaluations
        '''
        self.X_split = X_data

        if self.optimize_noise:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                return old_likelihood.full_NLML_grad(self.X_split, Y_data, self.kernel, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-3, jnp.ones_like(self.noise)*1e-3)
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise)
        else:
            def optim_fun(kernel_params):
                return old_likelihood.full_NLML_grad(self.X_split, Y_data, self.kernel, kernel_params, self.noise)  

            bounds = (1e-3, jnp.inf)  

            init_params = self.kernel_params  

        optimized_params = optimize(fun=optim_fun,
                                    params=init_params,
                                    bounds=bounds,
                                    method=self.optimize_method,
                                    callback=self.logger,
                                    jit_fun=True)
        
        if self.optimize_noise:
            self.kernel_params, self.noise = optimized_params
        else:
            self.kernel_params = optimized_params

        self.covar_module = jit(old_covar.full_covariance_matrix_grad)(self.X_split, Y_data, self.kernel, self.kernel_params, self.noise)

    def eval(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        '''evaluates the posterior mean and std for each point in X

        Parameters
        ----------
        X : ndarray
            shape (N, n_dims). Set of points for which to calculate the posterior means and stds

        Returns
        -------
        Posterior
            Posterior means and stds, [mean(x), std(x) for x in X]
        '''
        return jit(old_predict.full_predict_grad)(X, self.covar_module, self.X_split, self.kernel, self.kernel_params)

    def reset_params(self) -> None:
        '''Resets all kernel and noise parameters to initial values of log(2).
        '''
        self.kernel_params = jnp.ones(self.kernel.num_params)*jnp.log(2)
        self.noise = jnp.log(2)
    
    def tree_flatten(self):
        children = (self.kernel, self.kernel_params, self.noise, self.optimize_noise, 
                    self.optimize_method, self.logger, self.X_split, self.covar_module)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls(*children[:-2])
        new.X_split, new.covar_module = children[-2:]
        return new

@register_pytree_node_class
@dataclass
class ExactGPR:
    '''A full Gaussian Process regressor model.

    Parameters
    ----------
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : Union[float, ndarray], optional
        initial kernel parameters for the optimization
    noise : float, optional
        describes the noise present in the given input labels. Defaults to 1e-4
        If optimize_noise=True then this is taken as an initial parameter and optimized
    optimize_noise : bool, optional
        flag if noise parameter should also be optimized
    optimize_method : Optimizer, optional
        method to use in the optimizing process of the model parameters. By default the SLSQP method is used
    logger : Logger, optional
        If a Logger instance is given logs the results of the optimization procedure.
    '''
    kernel: Kernel
    kernel_params: Union[float, ndarray] = jnp.log(2)
    noise: float = 1e-4
    optimize_noise: bool = False
    optimize_method: Optimizer = Optimizer.SLSQP
    logger: Logger = None

    def __post_init__(self) -> None:
        if jnp.isscalar(self.kernel_params):
            self.kernel_params = jnp.ones(self.kernel.num_params)*self.kernel_params

        self.X_split = None
        self.covar_module = None

    def train(self, X_data: Tuple[ndarray, ndarray], Y_data: ndarray) -> None:
        '''Fits a full gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Tuple[ndarray, ndarray]
            Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
        Y_data : ndarray
            shape (n_function_evals + n_gradient_evals, ). Input labels representing noisy function evaluations
        '''
        self.X_split = X_data

        if self.optimize_noise:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                return old_likelihood.full_NLML(self.X_split, Y_data, self.kernel, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3)
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise)
        else:
            def optim_fun(kernel_params):
                return old_likelihood.full_NLML(self.X_split, Y_data, self.kernel, kernel_params, self.noise)  

            bounds = (1e-3, jnp.inf)  

            init_params = self.kernel_params  

        optimized_params = optimize(fun=optim_fun,
                                    params=init_params,
                                    bounds=bounds,
                                    method=self.optimize_method,
                                    callback=self.logger,
                                    jit_fun=True)
        
        if self.optimize_noise:
            self.kernel_params, self.noise = optimized_params
        else:
            self.kernel_params = optimized_params

        self.covar_module = jit(old_covar.full_covariance_matrix)(self.X_split, Y_data, self.kernel, self.kernel_params, self.noise)

    def eval(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        '''evaluates the posterior mean and std for each point in X

        Parameters
        ----------
        X : ndarray
            shape (N, n_dims). Set of points for which to calculate the posterior means and stds

        Returns
        -------
        Posterior
            Posterior means and stds, [mean(x), std(x) for x in X]
        '''
        return jit(old_predict.full_predict)(X, self.covar_module, self.X_split, self.kernel, self.kernel_params)

    def reset_params(self) -> None:
        '''Resets all kernel and noise parameters to initial values of log(2).
        '''
        self.kernel_params = jnp.ones(self.kernel.num_params)*jnp.log(2)
        self.noise = jnp.log(2)
    
    def tree_flatten(self):
        children = (self.kernel, self.kernel_params, self.noise, self.optimize_noise, 
                    self.optimize_method, self.logger, self.X_split, self.covar_module)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls(*children[:-2])
        new.X_split, new.covar_module = children[-2:]
        return new
    
@dataclass    
class SparseGPR:
    '''a sparse (FITC) Gaussian Process Regressor model.

    Parameters
    ----------
    X_ref : ndarray
        shape (n_referencepoints, n_dims). Reference points onto which the whole input dataset is projected.
    kernel : derived class from Kernel
        Kernel that describes the covariance between input points.
    kernel_params : Union[float, ndarray], optional
        initial kernel parameters for the optimization
    noise : float, optional
        describes the noise present in the given input labels. Defaults to 1e-4
        If optimize_noise=True then this is taken as an initial parameter and optimized
    optimize_noise : bool, optional
        flag if noise parameter should also be optimized
    optimize_ref : bool, optional
        flag if the positions of X_ref should be optimized
    ref_bounds : Tuple, optional
        Boundaries used for optimization of the reference positions, per default no bounds
    optimize_method : Optimizer, optional
        method to use in the optimizing process of the model parameters. By default the SLSQP method is used
    logger : Logger, optional
        If a Logger instance is given logs the results of the optimization procedure.
    '''
    X_ref: ndarray
    kernel: Kernel
    kernel_params: Union[float, ndarray] = jnp.log(2)
    noise: float = 1e-4
    optimize_noise: bool = False
    optimize_ref: bool = False
    ref_bounds: Tuple = (-jnp.inf, jnp.inf)
    optimize_method: Optimizer = Optimizer.SLSQP
    logger: Logger = None

    def __post_init__(self) -> None:
        if jnp.isscalar(self.kernel_params):
            self.kernel_params = jnp.ones(self.kernel.num_params)*self.kernel_params

        self.X_split = None
        self.covar_module = None

    def train(self, X_data: Tuple[ndarray, ndarray], Y_data: ndarray) -> None:
        '''Fits a sparse (FITC) gaussian process to the input data by optimizing the parameters of the model.

        Parameters
        ----------
        X_data : Tuple[ndarray, ndarray]
            Tuple( shape (n_function_evals, n_dims), shape (n_gradient_evals, n_dims) ). Input features at which the function and the gradient was evaluated
        Y_data : ndarray
            shape (n_function_evals + n_gradient_evals, ). Input labels representing noisy function evaluations
        '''
        self.X_split = X_data

        if self.optimize_noise and self.optimize_ref:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                X_ref = params[2]
                return old_likelihood.sparse_NLML(self.X_split, Y_data, X_ref, self.kernel, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3, jnp.ones_like(self.X_ref)*(self.ref_bounds[0]))
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf, jnp.ones_like(self.X_ref)*(self.ref_bounds[1]))

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise, self.X_ref)
        elif self.optimize_ref:
            def optim_fun(params):
                kernel_params = params[0]
                X_ref = params[1]
                return old_likelihood.sparse_NLML(self.X_split, Y_data, X_ref, self.kernel, kernel_params, self.noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.X_ref)*(-jnp.inf))
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.X_ref)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.X_ref)
        elif self.optimize_noise:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                return old_likelihood.sparse_NLML(self.X_split, Y_data, self.X_ref, self.kernel, kernel_params, noise)
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3)
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise)
        else:
            def optim_fun(params):
                return old_likelihood.sparse_NLML(self.X_split, Y_data, self.X_ref, self.kernel, params, self.noise)

            bounds = (1e-3, jnp.inf)  

            init_params = self.kernel_params

        optimized_params = optimize(fun=optim_fun,
                                    params=init_params,
                                    bounds=bounds,
                                    method=self.optimize_method,
                                    callback=self.logger,
                                    jit_fun=True)
        
        if self.optimize_noise and self.optimize_ref:
            self.kernel_params, self.noise, self.X_ref = optimized_params
        elif self.optimize_ref:
            self.kernel_params, self.X_ref = optimized_params
        elif self.optimize_noise:
            self.kernel_params, self.noise = optimized_params
        else:
            self.kernel_params = optimized_params

        self.covar_module = jit(old_covar.sparse_covariance_matrix)(self.X_split, Y_data, self.X_ref, self.kernel, self.kernel_params, self.noise)

    def eval(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        '''evaluates the posterior mean and std for each point in X

        Parameters
        ----------
        X : ndarray
            shape (N, n_features). Set of points for which to calculate the posterior means and stds

        Returns
        -------
        Posterior
            Posterior means and stds, [mean(x), std(x) for x in X]
        '''
        return jit(old_predict.sparse_predict)(X, self.covar_module, self.X_ref, self.kernel, self.kernel_params)