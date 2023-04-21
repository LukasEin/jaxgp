import jax.numpy as jnp
from jax import Array, jacfwd, jacrev
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class BaseKernel:
    '''Kernel base-class. Defines the needed derivatives of a kernel based 
    on the eval method. In each derived class the eval method must be overwritten.
    '''
    def __init__(self) -> None:
        self._df = jacrev(self.eval, argnums=1)
        self._ddf = jacfwd(self._df, argnums=0)

        self.num_params = None
    
    def eval(self, x1: Array, x2: Array, params: Array) -> float:
        '''covariance between two function evaluations at x1 and x2.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        params : Array
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
    
    def grad2(self, x1: Array, x2: Array, index: int, params: Array) -> float:
        '''covariance between a function evaluation at x1 and a derivative evaluation at x2.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a derivative evaluation.
        index : int
            derivative of the kernel is taken w.r.t. to x2[index]
        params : Array
            kernel parameters

        Returns
        -------
        float
            scalar value that describes the covariance between the points
        '''
        return self._df(x1, x2, params)[index]
    
    def jac(self, x1: Array, x2: Array, index_1: int, index_2: int, params: Array) -> float:
        '''double derivative of the Kernel w.r.t. x1[index_1] and x2[index_2]

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a derivative evaluation
        x2 : Array
            shape (n_features, ). Corresponds to a derivative evaluation
        index_1 : int
            one partial derivative of the kernel is taken w.r.t. x1[index1]
        index_2 : int
            the other partial derivative of the kernel is taken w.r.t. x2[index2]
        params : Array
            kernel parameters

        Returns
        -------
        float
            scalar value that describes the covariance between the points
        '''
        return self._ddf(x1, x2, params)[index_1, index_2]

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
    '''
    def __init__(self, num_params = 2):
        '''
        Parameters
        ----------
        num_params : int, optional
            by default 2, if changed must be set to n_features + 1, according to the input data.
        '''
        super().__init__()

        self.num_params = num_params
    
    def eval(self, x1, x2, params):
        '''covariance between two function evaluations at x1 and x2 
        according to the RBF kernel.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        params : Array
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
    '''
    def __init__(self, num_params=2):
        '''
        Parameters
        ----------
        num_params : int, optional
            by default 2, if changed must be set to n_features + 1, according to the input data.
        '''
        super().__init__()

        self.num_params = num_params

    def eval_func(self, x1, x2, params=(0.0, 1.0)):
        '''covariance between two function evaluations at x1 and x2 
        according to the Linear (dot-product) kernel.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        params : Array
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
    '''
    def __init__(self, num_params = 3):
        '''
        Parameters
        ----------
        num_params : int, optional
            by default 2, if changed must be set to n_features + 1, according to the input data.
        '''
        super().__init__()

        self.num_params = num_params
    
    def eval(self, x1, x2, params):
        '''covariance between two function evaluations at x1 and x2 
        according to the RBF kernel.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        params : Array
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
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

        self.num_params = kernel_1.num_params + kernel_2.num_params

    def eval_func(self, x1, x2, params):
        '''covariance between two function evaluations at x1 and x2 
        according to the sum of two kernels.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        params : Array
            shape (num_params, ). kernel parameters, parameters are split 
            according to the nmber of parameters each of the summed kernels has.

        Returns
        -------
        float
            Scalar value that describes the covariance between the points.
        '''
        return self.kernel_1.eval(x1, x2, params[:self.kernel_1.num_params]) + self.kernel_2.eval(x1, x2, params[self.kernel_1.num_params:])
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.kernel_1, self.kernel_2), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)

@register_pytree_node_class
class ProductKernel(BaseKernel):
    '''a wrapper that supplies multiplying two kernels
    '''
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

        self.num_params = kernel_1.num_params + kernel_2.num_params

    def eval_func(self, x1, x2, params):
        '''covariance between two function evaluations at x1 and x2 
        according to the product of two kernels.

        Parameters
        ----------
        x1 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        x2 : Array
            shape (n_features, ). Corresponds to a function evaluation.
        params : Array
            shape (num_params, ). kernel parameters, parameters are split 
            according to the nmber of parameters each of the summed kernels has.

        Returns
        -------
        float
            Scalar value that describes the covariance between the points.
        '''
        return self.kernel_1.eval(x1, x2, params[:self.kernel_1.num_params]) * self.kernel_2.eval(x1, x2, params[self.kernel_1.num_params:])
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.kernel_1, self.kernel_2), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)