from jax import grad
import jax.numpy as jnp
import numpy as np

class RBF_ND:
    def __init__(self,length_scale=1.0,coeff=1.0):
        '''
            length_scale.shape = (n_features,) as given in the evaluations 
                              or scalar
        '''
        self.length_scale = length_scale
        self.coeff = coeff
        
        self.df = grad(self.eval_func, argnums=1)
        self.ddf = grad(self.df)

    def eval_func(self, x1, x2):
        '''
            returns RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        diff = (x1 - x2) / self.length_scale

        return self.coeff * jnp.exp(-0.5 * jnp.dot(diff, diff))
    
    def eval_dx2(self, x1, x2):
        '''
            returns d/dx1 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        return self.df(x1, x2)
    
    def eval_ddx1x2(self, x1, x2):
        '''
            returns dd/dx1dx2 RBF(x1, x2)
            x1.shape = (n_features,)
            x2.shape = (n_features,)
            length_scale.shape = (n_features,) or scalar
        '''
        return self.ddf(x1, x2)

class RBF_1D:
    def __init__(self,length_scale=1.0,coeff=1.0):
        self.length_scale = length_scale
        self.coeff = coeff

    def eval(self,x1,x2):
        return self.coeff*np.exp(-0.5/self.length_scale**2 * (x1-x2)**2)
    
    def dx2(self,x1,x2):
        return (x1-x2)/self.length_scale**2 * self.eval(x1,x2)
    
    def ddx1x2(self,x1,x2):
        return (1 - (x1-x2)**2 / self.length_scale**2) / self.length_scale**2 * self.eval(x1,x2)