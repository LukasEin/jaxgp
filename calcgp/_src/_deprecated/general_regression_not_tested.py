from dataclasses import dataclass
from typing import Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.numpy import ndarray

from ..kernels import Kernel
from ..logger import Logger
from ..covar.covar import Prior
from ..likelihood import NegativeLogMarginalLikelihood
from ..posterior import Posterior
from ..regression.optimizer import OptimizerFlags, OptimizerTypes, parameter_optimize


@dataclass
class CalcGPR:
    prior_dist: Prior
    posterior_dist: Posterior
    nlml: NegativeLogMarginalLikelihood
    kernel: Kernel
    kernel_params: Union[float, ndarray] = jnp.log(2)
    noise: Union[float, ndarray] = 1e-4
    Xref: ndarray = None
    optim_method: OptimizerTypes = OptimizerTypes.SLSQP
    optim_flags: OptimizerFlags = OptimizerFlags()
    ref_bounds: Tuple = (-jnp.inf, jnp.inf)
    logger: Logger = None



    def __post_init__(self):
        if jnp.isscalar(self.kernel_params):
            self.kernel_params = jnp.ones(self.kernel.num_params)*self.kernel_params

        self.is_sparse = self.prior_dist.sparse
        if self.is_sparse != self.nlml.sparse or self.is_sparse != self.posterior_dist.sparse:
            raise ValueError(f"Excpected sparse flag to be the same everywhere, got\
                             prior={self.prior_dist.sparse}, \
                             nlml={self.nlml.sparse}, \
                             posterior={self.posterior_dist.sparse}!")
        
        if self.is_sparse and self.Xref is None:
            raise ValueError("If a sparse model is chosen Xref must be given!")

        if self.prior_dist.mode != self.posterior_dist.prior_mode:
            raise ValueError(f"Expected mode of the prior to be the same everywhere, got\
                             prior_dist={self.prior_dist.mode}, \
                             post_dist={self.posterior_dist.prior_mode}!")

        self.X_data = None
        self.prior_result = None



    def train(self, X_train, Y_train):
        prior_func = self.prior_dist()
        nlml_func = self.nlml()

        # --------------- SPARSE ---------------
        if self.is_sparse:
            # --------------- K-N-R ---------------
            if self.optim_flags.optim_kernel and self.optim_flags.optim_noise and self.optim_flags.optim_Xref:
                def optim_fun(params):
                    kernel_params = params[0]
                    noise = params[1]
                    Xref = params[2]
                    return nlml_func(prior_func(X_train, Y_train, Xref, self.kernel, kernel_params, noise))
                
                lb = (jnp.ones_like(self.kernel_params)*1e-3, jnp.ones_like(self.noise)*1e-3, jnp.ones_like(self.X_ref)*(self.ref_bounds[0]))
                ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf, jnp.ones_like(self.X_ref)*(self.ref_bounds[1]))

                bounds = (lb, ub)

                init_params = (self.kernel_params, self.noise, self.Xref)
            # --------------- K-N ---------------
            elif self.optim_flags.optim_kernel and self.optim_flags.optim_noise:
                def optim_fun(params):
                    kernel_params = params[0]
                    noise = params[1]
                    return nlml_func(prior_func(X_train, Y_train, self.Xref, self.kernel, kernel_params, noise))
            
                lb = (jnp.ones_like(self.kernel_params)*1e-3, jnp.ones_like(self.noise)*1e-3)
                ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

                bounds = (lb, ub)

                init_params = (self.kernel_params, self.noise)
            # --------------- K-R ---------------
            elif self.optim_flags.optim_kernel and self.optim_flags.optim_Xref:
                def optim_fun(params):
                    kernel_params = params[0]
                    Xref = params[1]
                    return nlml_func(prior_func(X_train, Y_train, Xref, self.kernel, kernel_params, self.noise))
                
                lb = (jnp.ones_like(self.kernel_params)*1e-3, jnp.ones_like(self.X_ref)*(self.ref_bounds[0]))
                ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.X_ref)*(self.ref_bounds[1]))

                bounds = (lb, ub)

                init_params = (self.kernel_params, self.Xref)
            # --------------- N-R ---------------
            elif self.optim_flags.optim_noise and self.optim_flags.optim_Xref:
                def optim_fun(params):
                    noise = params[0]
                    Xref = params[1]
                    return nlml_func(prior_func(X_train, Y_train, Xref, self.kernel, self.kernel_params, noise))
                
                lb = (jnp.ones_like(self.noise)*1e-3, jnp.ones_like(self.X_ref)*(self.ref_bounds[0]))
                ub = (jnp.ones_like(self.noise)*jnp.inf, jnp.ones_like(self.X_ref)*(self.ref_bounds[1]))

                bounds = (lb, ub)

                init_params = (self.noise, self.Xref)
            # --------------- K ---------------
            elif self.optim_flags.optim_kernel:
                def optim_fun(params):
                    return nlml_func(prior_func(X_train, Y_train, self.Xref, self.kernel, params, self.noise))
            
                bounds = (1e-3, jnp.inf)  

                init_params = self.kernel_params
            # --------------- N ---------------
            elif self.optim_flags.optim_noise:
                def optim_fun(params):
                    return nlml_func(prior_func(X_train, Y_train, self.Xref, self.kernel, self.kernel_params, params))
            
                bounds = (1e-3, jnp.inf)  

                init_params = self.noise
            # --------------- R ---------------
            elif self.optim_flags.optim_Xref:
                def optim_fun(params):
                    Xref = params[0]
                    return nlml_func(prior_func(X_train, Y_train, Xref, self.kernel, self.kernel_params, self.noise))
                
                lb = (jnp.ones_like(self.X_ref)*(self.ref_bounds[0]))
                ub = (jnp.ones_like(self.X_ref)*(self.ref_bounds[1]))

                bounds = (lb, ub)

                init_params = (self.noise, self.Xref)
        # --------------- FULL ---------------
        else:
            # --------------- K-N ---------------
            if self.optim_flags.optim_kernel & self.optim_flags.optim_noise:
                def optim_fun(params):
                    kernel_params = params[0]
                    noise = params[1]
                    return nlml_func(prior_func(X_train, Y_train, self.kernel, kernel_params, noise))
            
                lb = (jnp.ones_like(self.kernel_params)*1e-3, jnp.ones_like(self.noise)*1e-3)
                ub = (jnp.ones_like(self.kernel_params)*jnp.inf, jnp.ones_like(self.noise)*jnp.inf)

                bounds = (lb, ub)

                init_params = (self.kernel_params, self.noise)
            # --------------- K ---------------
            elif self.optim_flags.optim_kernel:
                def optim_fun(params):
                    return nlml_func(prior_func(X_train, Y_train, self.kernel, params, self.noise))
            
                bounds = (1e-3, jnp.inf)  

                init_params = self.kernel_params
            # --------------- N ---------------
            elif self.optim_flags.optim_noise:
                def optim_fun(params):
                    return nlml_func(prior_func(X_train, Y_train, self.kernel, self.kernel_params, params))
            
                bounds = (1e-3, jnp.inf)  

                init_params = self.noise
        
        optimized_params = parameter_optimize(fun=optim_fun,
                                              params=init_params,
                                              bounds=bounds,
                                              method=self.optimize_method,
                                              callback=self.logger,
                                              jit_fun=True)
        
        # --------------- SPARSE ---------------
        if self.is_sparse:
            # --------------- K-N-R ---------------
            if self.optim_flags.optim_kernel and self.optim_flags.optim_noise and self.optim_flags.optim_Xref:
                self.kernel_params, self.noise, self.Xref = optimized_params
            # --------------- K-N ---------------
            elif self.optim_flags.optim_kernel and self.optim_flags.optim_noise:
                self.kernel_params, self.noise = optimized_params
            # --------------- K-R ---------------
            elif self.optim_flags.optim_kernel and self.optim_flags.optim_Xref:
                self.kernel_params, self.Xref = optimized_params
            # --------------- N-R ---------------
            elif self.optim_flags.optim_noise and self.optim_flags.optim_Xref:
                self.noise, self.Xref = optimized_params
            # --------------- K ---------------
            elif self.optim_flags.optim_kernel:
                self.kernel_params = optimized_params
            # --------------- N ---------------
            elif self.optim_flags.optim_noise:
                self.noise = optimized_params
            # --------------- R ---------------
            elif self.optim_flags.optim_Xref:
                self.Xref = optimized_params
        # --------------- FULL ---------------
        else:
            # --------------- K-N ---------------
            if self.optim_flags.optim_kernel & self.optim_flags.optim_noise:
                self.kernel_params, self.noise = optimized_params
            # --------------- K ---------------
            elif self.optim_flags.optim_kernel:
                self.kernel_params = optimized_params
            # --------------- N ---------------
            elif self.optim_flags.optim_noise:
                self.noise = optimized_params

        if self.is_sparse:
            self.prior = jit(prior_func)(X_train, Y_train, self.Xref, self.kernel, self.kernel_params, self.noise)
        else:
            self.prior = jit(prior_func)(X_train, Y_train, self.kernel, self.kernel_params, self.noise)
        self.X_data = X_train



    def predict(self, X_test):
        post_func = self.posterior_dist()

        return jit(post_func)(X_test, self.prior, self.Xref, self.kernel, self.kernel_params)