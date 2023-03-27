from jax import grad, jacrev, jacfwd, vmap
import jax.numpy as jnp
from functools import partial

class BaseKernel:
    def __init__(self) -> None:
        pass

    def __add__(self, other):
        return SumKernel(self, other)
    
    def __mul__(self, other):
        return ProductKernel(self, other)

class RBF(BaseKernel):
    def __init__(self,length_scale=1.0,coeff=1.0):
        '''
            length_scale.shape = (n_features,) as given in the evaluations 
                              or scalar
        '''
        super().__init__()
        self.length_scale = length_scale
        self.coeff = coeff
        
        self._df = jacrev(self.eval_func, argnums=1)
        self._ddf = jacfwd(self._df)
    
    def eval_func(self, x1, x2):
        '''
            returns RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        diff = (x1 - x2) / self.length_scale
        return self.coeff * jnp.exp(-0.5 * jnp.dot(diff, diff))
    
    def eval_dx2(self, x1, x2, index):
        '''
            returns d/dx1 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar

            0 >= index (int) < n_features
                error if not given
        '''
        # if index is None:
        #     raise ValueError("Index missing!")
        
        return self._df(x1, x2)[index]
    
    def eval_ddx1x2(self, x1, x2, index_1, index_2):
        '''
            returns dd/dx1dx2 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
            
            0 >= index_i (int) < n_features if given
                if both are None the full Jacobian is returned 
                if only one is None the corresponding row/colums is returned
        '''
        # if index_1 is None or index_2 is None:
        #     raise ValueError("Indices missing!")
        
        return self._ddf(x1, x2)[index_1, index_2]
    
class Linear(BaseKernel):
    def __init__(self, offset=0.0):
        super().__init__()
        self.offset = offset

        self._df = jacrev(self.eval_func, argnums=1)
        self._ddf = jacfwd(self._df)

    def eval_func(self, x1, x2):
        '''
            returns Linear(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
        '''
        return jnp.inner(x1, x2) + self.offset
    
    def eval_dx2(self, x1, x2, index=None):
        '''
            returns d/dx1 Linear(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)

            0 >= index (int) < n_features
                error if not given
        '''
        if index is None:
            raise ValueError("Index missing!")
        
        return self._df(x1, x2)[index]
    
    def eval_ddx1x2(self, x1, x2, index_1 = None, index_2 = None):
        '''
            returns dd/dx1dx2 Linear(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            
            0 >= index_i (int) < n_features
                error if not both given
        '''
        if index_1 is None or index_2 is None:
            raise ValueError("Indices missing!")
        
        return self._ddf(x1, x2)[index_1, index_2]

class SumKernel(BaseKernel):
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def eval_func(self, x1, x2):
        '''
            returns Kernel1(x1, x2) + Kernel2(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        return self.kernel_1.eval_func(x1, x2) + self.kernel_2.eval_func(x1, x2)
    
    def eval_dx2(self, x1, x2, index):
        '''
            returns d/dx1 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar

            0 >= index (int) < n_features
                error if not given
        '''
        return self.kernel_1.eval_dx2(x1, x2, index) + self.kernel_2.eval_dx2(x1, x2, index)
    
    def eval_ddx1x2(self, x1, x2, index_1, index_2):
        '''
            returns dd/dx1dx2 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
            
            0 >= index_i (int) < n_features if given
                if both are None the full Jacobian is returned 
                if only one is None the corresponding row/colums is returned
        '''
        return self.kernel_1.eval_ddx1x2(x1, x2, index_1, index_2) + self.kernel_2.eval_ddx1x2(x1, x2, index_1, index_2)
    
class ProductKernel(BaseKernel):
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def eval_func(self, x1, x2):
        '''
            returns Kernel1(x1, x2) + Kernel2(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        return self.kernel_1.eval_func(x1, x2) * self.kernel_2.eval_func(x1, x2)
    
    def eval_dx2(self, x1, x2, index):
        '''
            returns d/dx1 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar

            0 >= index (int) < n_features
                error if not given
        '''
        return self.kernel_1.eval_dx2(x1, x2, index) * self.kernel_2.eval_dx2(x1, x2, index)
    
    def eval_ddx1x2(self, x1, x2, index_1, index_2):
        '''
            returns dd/dx1dx2 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
            
            0 >= index_i (int) < n_features if given
                if both are None the full Jacobian is returned 
                if only one is None the corresponding row/colums is returned
        '''
        return self.kernel_1.eval_ddx1x2(x1, x2, index_1, index_2) * self.kernel_2.eval_ddx1x2(x1, x2, index_1, index_2)