import jax.numpy as jnp
from jax.scipy.linalg import solve as scipysolve
from jax import jit
from functools import partial

def linsolve(A, x):
    return A._solve(x)

class LinearOperatorPPA:
    def __init__(self, K_MN, K_MM, diag) -> None:
        self.K_MN = K_MN
        self.K_MM = K_MM
        self.diag = diag

    def reduce_vector(self, vector):
        return self.K_MN@vector
    
    def solve(self, vector):
        if vector.shape[0] == self.K_MN.shape[1]:
            return self._solveN_pure(self.K_MN, self.K_MM, self.diag, vector)
        elif vector.shape[0] == self.K_MM.shape[1]:
            return self._solveM_pure(self.K_MN, self.K_MM, self.diag, vector)
    
    @partial(jit, static_argnums=(0,))
    def _solveM_pure(self, K_MN, K_MM, diag, vector):
        diag_inds = jnp.diag_indices(K_MM)
        invert = K_MM.at[diag_inds].add(diag**2) + K_MN@K_MN.T
        return scipysolve(invert, vector)
    
    @partial(jit, static_argnums=(0,))
    def _solveN_pure(self, K_MN, K_MM, diag, vector):
        diag_inds = jnp.diag_indices(K_MM.shape[1])
        invert = K_MM.at[diag_inds].add(diag**2) + K_MN@K_MN.T
        return scipysolve(invert, K_MN@vector)

class LinearOperatorNY:
    def __init__(self, K_MN, K_MM, diag) -> None:
        self.K_MN = K_MN
        self.K_MM = K_MM
        self.diag = diag

    def __init__(self, other: LinearOperatorPPA) -> None:
        self.K_MN = other.K_MN
        self.K_MM = other.K_MM
        self.diag = other.diag

    def solve(self, vector):
        return self._solve_pure(self.K_MN, self.K_MM, self.diag, vector)
    
    @partial(jit, static_argnums=(0,))
    def _solve_pure(self, K_MN, K_MM, diag, vector):
        diag_inds = jnp.diag_indices(K_MM.shape[1])
        invert = K_MM.at[diag_inds].add(diag**2) + K_MN@K_MN.T
        diag_inds = jnp.diag_indices(K_MN.shape[1])
        return -K_MN.T@scipysolve(invert, K_MN@vector).at[diag_inds].add(-1.0) / diag**2
    
    def logdet(self):
        return self._log_pure(self.K_MN, self.K_MM, self.diag)

    @partial(jit, static_argnums=(0,))
    def _logdet_pure(self, K_MN, K_MM, diag):
        diag_inds = jnp.diag_indices(K_MN.shape[1])
        return jnp.linalg.slogdet(K_MN.T@scipysolve(K_MM, K_MN).at[diag_inds].add(diag**2))[1]
    
class LinearOperatorFull:
    def __init__(self, K_NN, diag) -> None:
        diag_inds = jnp.diag_indices(K_NN.shape[1])
        self.K_NN = K_NN.at[diag_inds].add(diag**2)