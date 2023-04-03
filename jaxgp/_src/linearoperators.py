import jax.numpy as jnp
from jax.scipy.linalg import solve

class LinearOperatorPPA():
    def __init__(self, K_MN, K_MM, diag) -> None:
        self.K_MN = K_MN
        self.K_MM = K_MM
        self.diag = diag

    def reduce_vector(self, vector):
        return self.K_MN@vector
    
    def solveM(self, vector):
        return self._solveM(self.K_MN, self.K_MM, self.diag, vector)
    
    
    def _solveM(self, K_MN, K_MM, diag, vector):
        diag_inds = jnp.diag_indices(K_MM)
        invert = K_MM.at[diag_inds].add(diag) + K_MN@K_MN.T
        return solve(invert, vector)