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
    fitc_diag: ndarray

    def __post_init__(self):
        diag = jnp.diag_indices(len(self.K_MM))
        self.K_inv_cho = jsp.linalg.cho_factor((self.K_MM + self.K_NM.T@jnp.diag(1 / self.fitc_diag)@self.K_NM))#.at[diag].add(5e-2))

    def logdet(self):
        _, logdet_MM = jnp.linalg.slogdet(self.K_MM)
        logdet_fitc = jnp.sum(jnp.log(self.fitc_diag))
        K_diag = jnp.diag(self.K_inv_cho[0])
        logdet_K_inv = 2*jnp.sum(jnp.log(K_diag))

        return logdet_K_inv + logdet_fitc - logdet_MM
    
    def contract(self, left, right):
        complex_part = ((left / self.fitc_diag)@self.K_NM)@jsp.linalg.cho_solve(self.K_inv_cho,self.K_NM.T@(right / self.fitc_diag))
        easy_part = left@(right / self.fitc_diag)

        return easy_part - complex_part
    
    def mean():
        raise NotImplementedError

    def std():
        raise NotImplementedError
    
    def tree_flatten(self):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return ((self.K_NM, self.K_MM, self.fitc_diag), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        '''necessary for this class to be an argument in a jitted function (jax)
        '''
        return cls(*children)