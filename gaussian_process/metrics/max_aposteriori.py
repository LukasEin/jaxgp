import jax.numpy as jnp
from jax.scipy.linalg import solve
from functools import partial
from jax import jit, grad
from jax.scipy.stats import expon

def Zero():
    f = lambda x: 0.0
    return f

def Positive():
    f = lambda x: jnp.where(x > 0.0, 0.0, -jnp.inf)
    return f

def Greater(val):
    f = lambda x: jnp.where(x > val, 0.0, -jnp.inf)
    return f

def Expon(val):
    f = lambda x: jnp.where(x > 0.0, expon.logpdf(x, 0.0, val), -jnp.inf) 
    return f

def lognormal_prior(x,std):
    return -jnp.log(x*std) - 0.5 * (jnp.log(x))**2 / (std)**2

class MaximumAPosteriori:
    def __init__(self, kernel_constraint=Positive(), noise_constraint=Greater(1e-4),*, noise_prior=None) -> None:
        '''
            If a noise prior is given it must be in log form and return -jnp.inf for noise values < 0
        '''
        self.kernel_constraint = kernel_constraint
        if noise_prior is None:
            self.noise_constraint = noise_constraint
        else:
            self.noise_constraint = noise_prior
    
    @partial(jit, static_argnums=(0,))
    def forward(self, params, fitmatrix, fitvector):
        '''
            Does not calculate the full log Maximum a Posteriori 
            but just the parts that matter for the derivative.
        '''
        _, logdet = jnp.linalg.slogdet(fitmatrix)
        fitvector = fitvector.reshape(-1)
        mle = -0.5*(logdet + fitvector.T@solve(fitmatrix,fitvector, assume_a="pos"))
        prob_noise = self.noise_constraint(params[0])
        prob_kernel = jnp.sum(self.kernel_constraint(params[1:]))

        return mle + prob_noise + prob_kernel