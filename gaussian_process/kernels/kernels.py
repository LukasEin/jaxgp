from jax import grad, jacrev, jacfwd, vmap
import jax.numpy as jnp
from functools import partial

class BaseKernel:
    def __init__(self) -> None:
        self._df = jacrev(self.eval_func, argnums=1)
        self._ddf = jacfwd(self._df, argnums=0)

    def __add__(self, other):
        return SumKernel(self, other)
    
    def __mul__(self, other):
        return ProductKernel(self, other)

    def eval_func(self, x1, x2, params):
        raise NotImplementedError("Class deriving from BaseKernel has not implemented the method eval_func!")
    
    def eval_dx2(self, x1, x2, index, params):
        '''
            returns the derivative of the Kernel w.r.t x2[index]
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar

            0 >= index (int) < n_features
                error if not given
        '''
        # if index is None:
        #     raise ValueError("Index missing!")
        return self._df(x1, x2, params)[index]
    
    def eval_ddx1x2(self, x1, x2, index_1, index_2, params):
        '''
            returns the double derivative of the Kernel w.r.t. x1[index_1] and x2[index_2]
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
            
            0 >= index_i (int) < n_features if given
                if both are None the full Jacobian is returned 
                if only one is None the corresponding row/colums is returned
        '''
        # if index_1 is None or index_2 is None:
        #     raise ValueError("Indices missing!")
        return self._ddf(x1, x2, params)[index_1, index_2]

class RBF(BaseKernel):
    def __init__(self):
        '''
            length_scale.shape = (n_features,) as given in the evaluations 
                              or scalar
        '''
        super().__init__()
    
    def eval_func(self, x1, x2, params: tuple = (1.0, 1.0)):
        '''
            returns RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            params.shape = (1 + 1,) or (1 + n_features,)
                the first value describes the size coefficient in front,
                the rest the length_scale
        '''
        diff = (x1 - x2) / params[1:]
        return params[0] * jnp.exp(-0.5 * jnp.dot(diff, diff))
    
class Linear(BaseKernel):
    def __init__(self):
        super().__init__()

    def eval_func(self, x1, x2, params: tuple = (1.0, 0.0)):
        '''
            returns Linear(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            params.shape = (1 + 1,)
                the first value describes the additive offset,
                the second the length_scale
        '''
        return jnp.inner(x1, x2) * params[1] + params[0]

class SumKernel(BaseKernel):
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def eval_func(self, x1, x2, params: tuple):
        '''
            returns Kernel1(x1, x2) + Kernel2(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            params (tuple) 
        '''
        return self.kernel_1.eval_func(x1, x2, params[0]) + self.kernel_2.eval_func(x1, x2, params[1])
    
class ProductKernel(BaseKernel):
    def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
        super().__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def eval_func(self, x1, x2, params: tuple):
        '''
            returns Kernel1(x1, x2) + Kernel2(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        return self.kernel_1.eval_func(x1, x2, params[0]) * self.kernel_2.eval_func(x1, x2, params[1])

# class BaseKernel:
#     def __init__(self) -> None:
#         self._df = jacrev(self.eval_func, argnums=1)
#         self._ddf = jacfwd(self._df, argnums=0)

#     def __add__(self, other):
#         return SumKernel(self, other)
    
#     def __mul__(self, other):
#         return ProductKernel(self, other)

#     def eval_func(self, x1, x2):
#         raise NotImplementedError("Class deriving from BaseKernel has not implemented the method eval_func!")
    
#     def eval_dx2(self, x1, x2, index):
#         '''
#             returns the derivative of the Kernel w.r.t x2[index]
#             x1.shape = (n_features,)
#             x2.shape = (n_features,)
#             length_scale.shape = (n_features,) or scalar

#             0 >= index (int) < n_features
#                 error if not given
#         '''
#         # if index is None:
#         #     raise ValueError("Index missing!")
#         return self._df(x1, x2)[index]
    
#     def eval_ddx1x2(self, x1, x2, index_1, index_2):
#         '''
#             returns the double derivative of the Kernel w.r.t. x1[index_1] and x2[index_2]
#             x1.shape = (n_features,)
#             x2.shape = (n_features,)
#             length_scale.shape = (n_features,) or scalar
            
#             0 >= index_i (int) < n_features if given
#                 if both are None the full Jacobian is returned 
#                 if only one is None the corresponding row/colums is returned
#         '''
#         # if index_1 is None or index_2 is None:
#         #     raise ValueError("Indices missing!")
#         return self._ddf(x1, x2)[index_1, index_2]

# class RBF(BaseKernel):
#     def __init__(self, length_scale=1.0, pre_coeff=1.0):
#         '''
#             length_scale.shape = (n_features,) as given in the evaluations 
#                               or scalar
#         '''
#         self.length_scale = length_scale
#         self.pre_coeff = pre_coeff
#         super().__init__()
    
#     def eval_func(self, x1, x2):
#         '''
#             returns RBF(x1, x2)
#             x1.shape = (n_features,)
#             x2.shape = (n_features,)
#         '''
#         diff = (x1 - x2) / self.length_scale
#         return self.pre_coeff * jnp.exp(-0.5 * jnp.dot(diff, diff))
    
# class Linear(BaseKernel):
#     def __init__(self, length_scale=1.0, offset=0.0):
#         self.length_scale = length_scale
#         self.offset = offset
#         super().__init__()

#     def eval_func(self, x1, x2):
#         '''
#             returns Linear(x1, x2)
#             x1.shape = (n_features,)
#             x2.shape = (n_features,)
#             params.shape = (1 + 1,)
#         '''
#         return jnp.inner(x1, x2) / self.length_scale + self.offset

# class SumKernel(BaseKernel):
#     def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
#         super().__init__()
#         self.kernel_1 = kernel_1
#         self.kernel_2 = kernel_2

#     def eval_func(self, x1, x2):
#         '''
#             returns Kernel1(x1, x2) + Kernel2(x1, x2)
#             x1.shape = (n_features,)
#             x2.shape = (n_features,)
#             params (tuple) 
#         '''
#         return self.kernel_1.eval_func(x1, x2) + self.kernel_2.eval_func(x1, x2)
    
# class ProductKernel(BaseKernel):
#     def __init__(self, kernel_1 = BaseKernel(), kernel_2 = BaseKernel()) -> None:
#         super().__init__()
#         self.kernel_1 = kernel_1
#         self.kernel_2 = kernel_2

#     def eval_func(self, x1, x2):
#         '''
#             returns Kernel1(x1, x2) + Kernel2(x1, x2)
#             x1.shape = (n_features,)
#             x2.shape = (n_features,)
#             length_scale.shape = (n_features,) or scalar
#         '''
#         return self.kernel_1.eval_func(x1, x2) * self.kernel_2.eval_func(x1, x2)