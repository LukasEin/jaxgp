from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.numpy import ndarray
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class Kernel(ABC):
    '''Kernel base-class. Defines the needed derivatives of a kernel based 
    on the eval method. In each derived class the eval method must be overwritten.
    
    Parameters
    ----------
    num_params : int, optional
    '''
    num_params: int = 2
    
    @abstractmethod
    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        pass
    
    def grad2(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''gradient of the kernel w.r.t. x2

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a gradient evaluation.
        params : ndarray
            kernel parameters

        Returns
        -------
        ndarray
            shae (n_features,)
            vector value that describes the gradient of the kernel at points x1 and x2
        '''
        return jacrev(self.eval, argnums=1)(x1, x2, params)
    
    def jac(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''double gradient of the Kernel w.r.t. x1 and x2

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a gradient evaluation
        x2 : ndarray
            shape (n_features, ). Corresponds to a gradient evaluation
        params : ndarray
            kernel parameters

        Returns
        -------
        ndarray
            shape (n_features, n_features).
            matrix value that describes the "hessian" of the kernel at points x1 and x2
            (only the true hessian if x1 == x2)
        '''
        return jacfwd(jacrev(self.eval, argnums=1), argnums=0)(x1, x2, params)

    def __add__(self, other):
        return SumKernel(self, other)
    
    def __mul__(self, other):
        return ProductKernel(self, other)
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.num_params, ), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)

@register_pytree_node_class
@dataclass
class RBF(Kernel):
    '''Kernel based on radial basis function / gaussian

    Parameters
    ----------
    num_params : int, optional
        by default 2, if changed must be set to n_features + 1, according to the input data.
    '''
    num_params: int = 2
    
    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''covariance between two function evaluations at x1 and x2 
        according to the RBF kernel.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        params : ndarray
            shape (num_params, ). kernel parameters, the first is a multiplicative constant, 
                                the rest a scale parameter of the inputs

        Returns
        -------
        ndarray
            shape ()
            Scalar value that describes the kernel evaluation at points x1 and x2.
        '''
        # assert len(x1.shape) == 1 and len(x2.shape) == 1, f"Input points must all be 1-dimensional, got: {x1.shape}, {x2.shape}!"
        # assert len(params.shape) == 1 and (len(params) == 2 or len(params) == len(x1) + 1) , \
        #     f"Parameters must be 1-dimensional and of shape ({self.num_params},) got: {params.shape}!"

        diff = (x1 - x2) / params[1:]
        return params[0]*jnp.exp(-0.5 * jnp.dot(diff, diff))

@register_pytree_node_class  
@dataclass  
class Linear(Kernel):
    '''kernel based on the dot-product of the two input vectors

    Parameters
    ----------
    num_params : int, optional
        by default 2, if changed must be set to n_features + 1, according to the input data.
    '''
    num_params: int = 2

    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''covariance between two function evaluations at x1 and x2 
        according to the Linear (dot-product) kernel.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        params : ndarray
            shape (num_params, ). kernel parameters, the first is an additive constant, 
                                the rest a scale parameter of the inputs

        Returns
        -------
        ndarray
            shape ()
            Scalar value that describes the kernel evaluation at points x1 and x2.
        '''
        # assert len(x1.shape) == 1 and len(x2.shape) == 1, f"Input points must all be 1-dimensional, got: {x1.shape}, {x2.shape}!"
        # assert len(params.shape) == 1 and (len(params) == 2 or len(params) == len(x1) + 1) , \
        #     f"Parameters must be 1-dimensional and of shape ({self.num_params},) got: {params.shape}!"

        return jnp.inner(x1 * params[1:], x2) + params[0]
    
@register_pytree_node_class
@dataclass  
class Periodic(Kernel):
    '''Kernel based on radial basis function / gaussian
    
    Parameters
    ----------
    num_params : int, optional
        by default 2, if changed must be set to n_features + 1, according to the input data.
    '''
    num_params: int = 3
    
    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''covariance between two function evaluations at x1 and x2 
        according to the RBF kernel.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        params : ndarray
            shape (num_params, ). kernel parameters, the first is a multiplicative constant, 
                                the rest a scale parameter of the inputs

        Returns
        -------
        ndarray
            shape ()
            Scalar value that describes the kernel evaluation at points x1 and x2.
        '''
        # assert len(x1.shape) == 1 and len(x2.shape) == 1, f"Input points must all be 1-dimensional, got: {x1.shape}, {x2.shape}!"
        # assert len(params.shape) == 1 and (len(params) == 3 or len(params) == len(x1) + 2) , \
        #     f"Parameters must be 1-dimensional and of shape ({self.num_params},) got: {params.shape}!"

        periodic = jnp.sin(jnp.pi*(x1-x2)/params[2:])**2
        return params[0]*jnp.exp(-(2 / params[1]**2) * jnp.sum(periodic))

@register_pytree_node_class
@dataclass  
class SumKernel(Kernel):
    '''A wrapper that supplies the summing of two kernels
    '''
    left_kernel: Kernel
    right_kernel: Kernel
    num_params: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_params = self.left_kernel.num_params + self.right_kernel.num_params

    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''covariance between two function evaluations at x1 and x2 
        according to the sum of two kernels.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        params : ndarray
            shape (num_params, ). kernel parameters, parameters are split 
            according to the nmber of parameters each of the summed kernels has.

        Returns
        -------
        ndarray
            shape ()
            Scalar value that describes the kernel evaluation at points x1 and x2.
        '''
        return self.left_kernel.eval(x1, x2, params[:self.left_kernel.num_params]) + self.right_kernel.eval(x1, x2, params[self.left_kernel.num_params:])
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.left_kernel, self.right_kernel), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)

@register_pytree_node_class
@dataclass  
class ProductKernel(Kernel):
    '''a wrapper that supplies multiplying two kernels
    '''
    left_kernel: Kernel
    right_kernel: Kernel
    num_params: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_params = self.left_kernel.num_params + self.right_kernel.num_params

    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''covariance between two function evaluations at x1 and x2 
        according to the product of two kernels.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        params : ndarray
            shape (num_params, ). kernel parameters, parameters are split 
            according to the nmber of parameters each of the summed kernels has.

        Returns
        -------
        ndarray
            shape ()
            Scalar value that describes the kernel evaluation at points x1 and x2.
        '''
        return self.left_kernel.eval(x1, x2, params[:self.left_kernel.num_params]) * self.right_kernel.eval(x1, x2, params[self.left_kernel.num_params:])
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.left_kernel, self.right_kernel), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)