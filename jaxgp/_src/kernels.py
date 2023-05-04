import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.numpy import ndarray
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass, field


@register_pytree_node_class
@dataclass
class BaseKernel:
    '''Kernel base-class. Defines the needed derivatives of a kernel based 
    on the eval method. In each derived class the eval method must be overwritten.
    '''
    num_params: int = 0
    
    def eval(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
        '''covariance between two function evaluations at x1 and x2.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        params : ndarray
            kernel_parameters

        Returns
        -------
        float
            Scalar value that describes the covariance between the points.

        Raises
        ------
        NotImplementedError
            This method must be overwritten in all derived classes.
        '''
        raise NotImplementedError("Class deriving from BaseKernel has not implemented the method eval!")
    
    def grad2(self, x1: ndarray, x2: ndarray, params: ndarray) -> float:
        '''covariance between a function evaluation at x1 and a derivative evaluation at x2.

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : ndarray
            shape (n_features, ). Corresponds to a derivative evaluation.
        index : int
            derivative of the kernel is taken w.r.t. to x2[index]
        params : ndarray
            kernel parameters

        Returns
        -------
        float
            scalar value that describes the covariance between the points
        '''
        return jacrev(self.eval, argnums=1)(x1, x2, params)
    
    def jac(self, x1: ndarray, x2: ndarray, params: ndarray) -> float:
        '''double derivative of the Kernel w.r.t. x1[index_1] and x2[index_2]

        Parameters
        ----------
        x1 : ndarray
            shape (n_features, ). Corresponds to a derivative evaluation
        x2 : ndarray
            shape (n_features, ). Corresponds to a derivative evaluation
        index_1 : int
            one partial derivative of the kernel is taken w.r.t. x1[index1]
        index_2 : int
            the other partial derivative of the kernel is taken w.r.t. x2[index2]
        params : ndarray
            kernel parameters

        Returns
        -------
        float
            scalar value that describes the covariance between the points
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
class RBF(BaseKernel):
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
        float
            Scalar value that describes the covariance between the points.
        '''
        diff = (x1 - x2) / params[1:]
        return params[0]*jnp.exp(-0.5 * jnp.dot(diff, diff))

@register_pytree_node_class    
class Linear(BaseKernel):
    '''kernel based on the dot-product of the two input vectors
    Parameters
    ----------
    num_params : int, optional
        by default 2, if changed must be set to n_features + 1, according to the input data.
    '''
    num_params: int = 2

    def eval_func(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
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
        float
            Scalar value that describes the covariance between the points.
        '''
        return jnp.inner(x1 * params[1:], x2) + params[0]
    
@register_pytree_node_class
class Periodic(BaseKernel):
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
        float
            Scalar value that describes the covariance between the points.
        '''
        periodic = jnp.sin(jnp.pi*(x1-x2)/params[1])**2
        return params[0]*jnp.exp(-(2 / params[2]**2) * jnp.sum(periodic))

@register_pytree_node_class
class SumKernel(BaseKernel):
    '''A wrapper that supplies the summing of two kernels
    '''
    left_kernel: BaseKernel
    right_kernel: BaseKernel
    num_params: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_params = self.left_kernel.num_params + self.right_kernel.num_params

    def eval_func(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
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
        float
            Scalar value that describes the covariance between the points.
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
class ProductKernel(BaseKernel):
    '''a wrapper that supplies multiplying two kernels
    '''
    left_kernel: BaseKernel
    right_kernel: BaseKernel
    num_params: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_params = self.left_kernel.num_params + self.right_kernel.num_params

    def eval_func(self, x1: ndarray, x2: ndarray, params: ndarray) -> ndarray:
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
        float
            Scalar value that describes the covariance between the points.
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