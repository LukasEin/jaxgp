import jax.numpy as jnp
from jax.scipy.linalg import solve
from functools import partial
from jax import jit

from .linearoperators import linsolve, LinearOperatorNY

def Zero():
    f = lambda x: 0.0
    return f

class MaximumAPosteriori:
    def __init__(self, *, kernel_prior=Zero(), noise_prior=Zero()) -> None:
        '''
            If a noise prior is given it must be in log form and return -jnp.inf for noise values < 0
        '''
        self.kernel_prior = kernel_prior
        self.noise_prior = noise_prior
    
    @partial(jit, static_argnums=(0,))
    def forward(self, params, kernel_operator: LinearOperatorNY, Y_data):
        '''
            Does not calculate the full log Maximum a Posteriori 
            but just the parts that matter for the derivative.
        '''
        logdet = kernel_operator.logdet()
        Y_data = Y_data.reshape(-1)
        mle = -0.5*(logdet + Y_data.T@linsolve(kernel_operator,Y_data))#, assume_a="pos"))
        prob_noise = self.noise_prior(params[0])
        prob_kernel = jnp.sum(self.kernel_prior(params[1:]))

        return mle + prob_noise + prob_kernel