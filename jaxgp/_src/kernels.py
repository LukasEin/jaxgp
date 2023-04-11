from jax import jacrev, jacfwd
import jax.numpy as jnp

from jax import Array

class BaseKernel:
    '''A base class for all kernels that defines the derivatives of the eval method. The eval method must be overwritten in any derived classes.
    '''
    def __init__(self) -> None:
        self._df = jacrev(self.eval, argnums=1)
        self._ddf = jacfwd(self._df, argnums=0)

        self.num_params = 0

    def __add__(self, other):
        return SumKernel(self, other)
    
    def __mul__(self, other):
        return ProductKernel(self, other)

    def eval(self, x1: Array, x2: Array, params: Array):
        '''Kernel evaluation

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of first function evaluation
        x2 : Array
            (n_features, ), point of second function evaluation
        params : Array
            kernel parameters

        Raises
        ------
        NotImplementedError
            Any Kernel must override this method
        '''
        raise NotImplementedError("Class deriving from BaseKernel has not implemented the method eval!")
    
    def grad2(self, x1: Array, x2: Array, index: int, params: Array) -> float:
        '''derivative of the Kernel w.r.t x2[index]

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of function evaluation
        x2 : Array
            (n_features, ), point of derivative evaluation
        index : int
            index of x2 w.r.t. which the derivative is calculated
        params : Array
            kernel parameters

        Returns
        -------
        float
            covariance between a function evaluation and a derivative evaluation
        '''
        return self._df(x1, x2, params)[index]
    
    def jac(self, x1: Array, x2: Array, index1: int, index2: int, params: Array) -> float:
        '''derivative of the Kernel w.r.t x1 [index1] and x2[index2]

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of first derivative evaluation
        x2 : Array
            (n_features, ), point of second derivative evaluation
        index1 : int
            index of x1 w.r.t. which the derivative is calculated
        index2 : int
            index of x2 w.r.t. which the derivative is calculated
        params : Array
            kernel parameters

        Returns
        -------
        float
            covariance between two derivative evaluations
        '''
        return self._ddf(x1, x2, params)[index1, index2]

class RBF(BaseKernel):
    def __init__(self, num_params=1):
        super().__init__()

        self.num_params = num_params
    
    def eval(self, x1: Array, x2: Array, params: Array) -> float:
        '''RBF Kernel

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of first function evaluation
        x2 : Array
            (n_features, ), point of second function evaluation
        params : Array
            scalar or (n_features, ), kernel parameters

        Returns
        -------
        float
            covariance between two function evaluations
        '''
        diff = (x1 - x2) / params
        return jnp.exp(-0.5 * jnp.dot(diff, diff))
    
class Linear(BaseKernel):
    def __init__(self, num_params=2):
        super().__init__()

        self.num_params = num_params

    def eval(self, x1: Array, x2: Array, params: Array) -> float:
        '''Linear Kernel

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of first function evaluation
        x2 : Array
            (n_features, ), point of second function evaluation
        params : Array
            (1 + 1, ) or (1 + n_features, ), kernel parameters

        Returns
        -------
        float
            covariance between two function evaluations
        '''
        return jnp.inner(x1, x2) * params[1:] + params[0]

class SumKernel(BaseKernel):
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

        self.num_params = kernel_1.num_params + kernel_2.num_params

    def eval(self, x1: Array, x2: Array, params: Array) -> float:
        '''Sum of two Kernels

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of first function evaluation
        x2 : Array
            (n_features, ), point of second function evaluation
        params : Array
            (n_params1, n_params2), kernel parameters

        Returns
        -------
        float
            covariance between two function evaluations
        '''
        return self.kernel_1.eval(x1, x2, params[:self.kernel_1.num_params]) + self.kernel_2.eval(x1, x2, params[self.kernel_1.num_params:])
    
class ProductKernel(BaseKernel):
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

        self.num_params = kernel_1.num_params + kernel_2.num_params

    def eval(self, x1: Array, x2: Array, params: Array) -> float:
        '''Product of two Kernels

        Parameters
        ----------
        x1 : Array
            (n_features, ), point of first function evaluation
        x2 : Array
            (n_features, ), point of second function evaluation
        params : Array
            (n_params1, n_params2), kernel parameters

        Returns
        -------
        float
            covariance between two function evaluations
        '''
        return self.kernel_1.eval(x1, x2, params[:self.kernel_1.num_params]) * self.kernel_2.eval(x1, x2, params[self.kernel_1.num_params:])