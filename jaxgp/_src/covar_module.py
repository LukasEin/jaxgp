import jax.numpy as jnp
from jax.numpy import ndarray
from dataclasses import dataclass
from jax.scipy.linalg import solve
import jax.scipy as jsp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class SparseCovarModule:
    K_NM: ndarray
    K_MM: ndarray
    noise: float
    # diagonal: ndarray # for FITC

    def __post_init__(self):
        self.K_inv_cho = jsp.linalg.cho_factor(self.K_MM + self.K_NM.T@self.K_NM / self.noise**2)
        # self.K_inv_cho = jsp.linalg.cho_factor(self.noise**2 * self.K_MM + self.K_NM.T@self.K_NM)

    def logdet(self):
        _, logdet_MM = jnp.linalg.slogdet(self.K_MM)
        logdet_noise = 2*len(self.K_NM)*jnp.log(self.noise)
        diag = jnp.diag(self.K_inv_cho[0])
        logdet_K_inv = 2*jnp.sum(jnp.log(diag))

        return logdet_K_inv + logdet_noise - logdet_MM
    
    def contract(self, left, right):
        complex_part = (left@self.K_NM)@jsp.linalg.cho_solve(self.K_inv_cho,self.K_NM.T@right) / self.noise**2
        # complex_part = (left@self.K_NM)@jsp.linalg.cho_solve(self.K_inv_cho,self.K_NM.T@right)
        easy_part = left@right

        return (easy_part - complex_part) / self.noise**2
    
    def mean():
        raise NotImplementedError

    def std():
        raise NotImplementedError
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.K_NM, self.K_MM, self.noise), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)