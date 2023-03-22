from jax import grad, jacrev
import jax.numpy as jnp

class RBF:
    def __init__(self,length_scale=1.0,coeff=1.0):
        '''
            length_scale.shape = (n_features,) as given in the evaluations 
                              or scalar
        '''
        self.length_scale = length_scale
        self.coeff = coeff
        
        self._df = grad(self.eval_func, argnums=1)
        self._ddf = jacrev(self._df)
    
    def eval_func(self, x1, x2):
        '''
            returns RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        diff = (x1 - x2) / self.length_scale


        return self.coeff * jnp.exp(-0.5 * jnp.dot(diff, diff))
    
    def eval_dx2(self, x1, x2, index=None):
        '''
            returns d/dx1 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar

            0 >= index (int) < n_features if given
                if None the full Hessian is returned 
        '''
        if index is None:
            return self._df(x1, x2)
        
        return self._df(x1,x2)[index]
    
    def eval_ddx1x2(self, x1, x2, index_1 = None, index_2 = None):
        '''
            returns dd/dx1dx2 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
            
            0 >= index_i (int) < n_features if given
                if both are None the full Jacobian is returned 
                if only one is None the corresponding row/colums is returned
        '''
        if index_1 is not None and index_2 is not None:
            return self._ddf(x1, x2)[index_1, index_2]
        
        if index_1 is None:
            return self._ddf(x1, x2)[:, index_2]
        
        if index_2 is None:
            return self._ddf(x1, x2)[index_1]
        
        return self._ddf(x1, x2)