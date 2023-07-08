from dataclasses import dataclass
from typing import Tuple, Union
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import jit
from jax.numpy import ndarray

from ..kernels import Kernel
from ..logger import Logger
from ..covar.covar import DataMode, Prior
from ..likelihood import NegativeLogMarginalLikelihood
from ..posterior import Posterior
from .optimizer import OptimizerTypes, parameter_optimize


@dataclass
class SparseGPRBase(ABC):
    kernel: Kernel
    Xref: ndarray
    kernel_params: Union[float, ndarray] = jnp.log(2)
    noise: Union[float, ndarray] = 1e-2
    optim_method: OptimizerTypes = OptimizerTypes.SLSQP
    optim_noise: bool = False
    logger: Logger = None

    def __post_init__(self):
        if jnp.isscalar(self.kernel_params):
            self.kernel_params = jnp.ones(self.kernel.num_params)*self.kernel_params

        self.prior_result = None
        self.nlml = NegativeLogMarginalLikelihood(sparse=True)

    @abstractmethod
    def train(self, X_data, Y_data):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def _train(self, X_data, Y_data):
        prior_func = self.prior()
        nlml_func = self.nlml()

        if self.optimize_noise and self.optimize_ref:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                Xref = params[2]
                return nlml_func(prior_func(X_data, Y_data, Xref, self.kernel, kernel_params, noise))
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3, jnp.ones_like(self.Xref)*(self.ref_bounds[0]))
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf, jnp.ones_like(self.Xref)*(self.ref_bounds[1]))

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise, self.Xref)
        elif self.optimize_ref:
            def optim_fun(params):
                kernel_params = params[0]
                Xref = params[1]
                return nlml_func(prior_func(X_data, Y_data, Xref, self.kernel, kernel_params, self.noise))
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.Xref)*(-jnp.inf))
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.Xref)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.Xref)
        elif self.optimize_noise:
            def optim_fun(params):
                kernel_params = params[0]
                noise = params[1]
                return nlml_func(prior_func(X_data, Y_data, self.Xref, self.kernel, kernel_params, noise))
            
            lb = (jnp.ones_like(self.kernel_params)*1e-6, jnp.ones_like(self.noise)*1e-3)
            ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

            bounds = (lb, ub)

            init_params = (self.kernel_params, self.noise)
        else:
            def optim_fun(params):
                return nlml_func(prior_func(X_data, Y_data, self.Xref, self.kernel, params, self.noise))

            bounds = (1e-3, jnp.inf)  

            init_params = self.kernel_params

        optimized_params = parameter_optimize(fun=optim_fun,
                                              params=init_params,
                                              bounds=bounds,
                                              method=self.optim_method,
                                              callback=self.logger,
                                              jit_fun=True)
        
        if self.optimize_noise and self.optimize_ref:
            self.kernel_params, self.noise, self.Xref = optimized_params
        elif self.optimize_ref:
            self.kernel_params, self.Xref = optimized_params
        elif self.optimize_noise:
            self.kernel_params, self.noise = optimized_params
        else:
            self.kernel_params = optimized_params

        self.prior_result = jit(prior_func)(X_data, Y_data, self.Xref, self.kernel, self.kernel_params, self.noise)

    def _predict(self, X_test):
        post_func = self.posterior()

        return jit(post_func)(X_test, self.prior_result, self.kernel, self.kernel_params)
    


@dataclass
class SparseGradient(SparseGPRBase):
    def train(self, X_data, Y_data):
        if isinstance(X_data, Tuple):
            self.prior = Prior(sparse=True, mode=DataMode.MIX)
        else:
            self.prior = Prior(sparse=True, mode=DataMode.FUNC)

        return self._train(X_data, Y_data)
    
    def predict(self, X_test):
        self.posterior = Posterior(sparse=True, prior_mode=self.prior.mode, posterior_mode=DataMode.GRAD)

        return self._predict(X_test)
    


@dataclass
class SparseIntegral(SparseGPRBase):
    def train(self, X_data, Y_data):
        if isinstance(X_data, Tuple):
            self.prior = Prior(sparse=True, mode=DataMode.MIX)
        else:
            self.prior = Prior(sparse=True, mode=DataMode.GRAD)

        return self._train(X_data, Y_data)
    
    def predict(self, X_test):
        self.posterior = Posterior(sparse=True, prior_mode=self.prior.mode, posterior_mode=DataMode.FUNC)

        return self._predict(X_test)
    

    
@dataclass
class SparseFunction(SparseGPRBase):
    def train(self, X_data, Y_data):
        self.prior = Prior(sparse=True, mode=DataMode.FUNC)

        return self._train(X_data, Y_data)
    
    def predict(self, X_test):
        self.posterior = Posterior(sparse=True, prior_mode=self.prior.mode, posterior_mode=DataMode.FUNC)

        return self._predict(X_test)