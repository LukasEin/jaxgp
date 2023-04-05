import jax.numpy as jnp
from jax.scipy.linalg import solve as scipysolve
from jax import jit
from functools import partial

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class LinearOperatorPPA:
    def __init__(self, K_MN, K_MM, diag) -> None:
        self.K_MN = K_MN
        self.K_MM = K_MM
        self.diag = diag

    def reduce_vector(self, vector):
        return self.K_MN@vector
    
    def solve(self, vector):
        if vector.shape[0] == self.K_MN.shape[1]:
            return LinearOperatorPPA._solveN_pure(self.K_MN, self.K_MM, self.K_MM.shape[1], self.diag, vector)
        elif vector.shape[0] == self.K_MM.shape[1]:
            return LinearOperatorPPA._solveM_pure(self.K_MN, self.K_MM, self.K_MM.shape[1], self.diag, vector)
        
    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _solveM_pure(K_MN, K_MM, M, diag, vector):
        diag_inds = jnp.diag_indices(M)
        invert = K_MM.at[diag_inds].add(diag**2) + K_MN@K_MN.T
        return scipysolve(invert, vector)
    
    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _solveN_pure(K_MN, K_MM, M, diag, vector):
        diag_inds = jnp.diag_indices(M)
        invert = K_MM.at[diag_inds].add(diag**2) + K_MN@K_MN.T
        return scipysolve(invert, K_MN@vector)
    
    def tree_flatten(self):
        children = (self.K_MN, self.K_MM, self.diag)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class LinearOperatorNY:
    def __init__(self, K_MN, K_MM, diag) -> None:
        self.K_MN = K_MN
        self.K_MM = K_MM
        self.diag = diag

    def solve(self, vector):
        return LinearOperatorNY._solve_pure(self.K_MN, self.K_MM, self.K_MM.shape[1], self.diag, vector)
    
    def logdet(self):
        return LinearOperatorNY._logdet_pure(self.K_MN, self.K_MM, self.K_MN.shape[1], self.diag)
    
    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _solve_pure(K_MN, K_MM, M, diag, vector):
        diag_inds = jnp.diag_indices(M)
        invert = K_MM*diag**2 + K_MN@K_MN.T
        # invert = K_MM.at[diag_inds].add(diag**2) + K_MN@K_MN.T
        return (vector - K_MN.T@scipysolve(invert, K_MN@vector)) / diag**2

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _logdet_pure(K_MN, K_MM, N, diag):
        diag_inds = jnp.diag_indices(N)
        return jnp.linalg.slogdet((K_MN.T@scipysolve(K_MM, K_MN)).at[diag_inds].add(diag**2))[1]
    
    def tree_flatten(self):
        children = (self.K_MN, self.K_MM, self.diag)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
@register_pytree_node_class
class LinearOperatorFull:
    def __init__(self, K_NN, diag) -> None:
        self.K_NN = K_NN
        self.diag = diag

    def solve(self, x):
        return LinearOperatorFull._solve_pure(self.K_NN, self.K_NN.shape[1], self.diag, x)
    
    def logdet(self):
        return LinearOperatorFull._logdet_pure(self.K_NN, self.K_NN.shape[1], self.diag)
    
    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _solve_pure(K_NN, N, diag, x):
        diag_inds = jnp.diag_indices(N)
        return scipysolve(K_NN.at[diag_inds].add(diag**2), x)
    
    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _logdet_pure(K_NN, N, diag):
        diag_inds = jnp.diag_indices(N)
        return jnp.linalg.slogdet(K_NN.at[diag_inds].add(diag**2))[1]
    
    def tree_flatten(self):
        children = (self.K_NN, self.diag)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def linsolve(A: LinearOperatorNY, x):
    return A.solve(x)