import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from typing import Tuple, Union, Callable
from jax import Array

def create_training_data_2D(seed: int, num_datapoints: int, ranges: Tuple, noise: Union[float, Array], test_function: Callable) -> Tuple[Array, Array]:
    '''creates training data for 2D functions

    Parameters
    ----------
    seed : int
        seed for random key
    num_datapoints : int
        number of datapoints to return
    ranges : Tuple
        ranges for the input space
    noise : Union[float, Array]
        noise for the observations
    test_function : Callable
        function to get the observations from

    Returns
    -------
    Tuple[Array, Array]
        tuple of training data. First arguments are the features, 
        second argument are the function and derivative observations
        shapes (n_samples, 2), (n_samples, 3)
    '''
    f = jit(vmap(test_function, in_axes=(0,)))
    df = jit(vmap(grad(test_function, argnums=0), in_axes=(0,)))

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    X1 = random.uniform(subkey, (num_datapoints,), minval=ranges[0], maxval=ranges[1])
    key, subkey = random.split(key)
    X2 = random.uniform(subkey, (num_datapoints,), minval=ranges[2], maxval=ranges[3])
    X = jnp.vstack((X1, X2)).T

    key, subkey = random.split(key)
    Y = f(X) + noise*random.normal(subkey, (num_datapoints,))
    key, subkey = random.split(key)
    dY = df(X) + noise*random.normal(subkey, (num_datapoints,2))

    Y = jnp.hstack((Y.reshape(-1,1), dY))

    return X, Y

def standard_parabola(x):
    '''range = (-5,5)
    '''
    return x[0]**2 + x[1]**2

def stretched_parabola(x):
    '''range = (-10,10)
    '''
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def easom(x):
    '''range = (-10,10)
    '''
    return -jnp.cos(x[0]) * jnp.cos(x[1]) * jnp.exp(-((x[0] - jnp.pi)**2 + (x[1] - jnp.pi)**2))

def ackley(x):
    '''range = (-5,5)
    '''
    return -20.0*jnp.exp(-0.2*jnp.sqrt(0.5*(x[0]**2 + x[1]**2))) - jnp.exp(0.5*(jnp.cos(2*jnp.pi * x[0]) + jnp.cos(2*jnp.pi*x[1]))) + jnp.e + 20

def himmelblau(x):
    '''range = (-5,5)
    '''
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

def holder(x):
    '''range = (-10,10)
    '''
    return -jnp.abs(jnp.sin(x[0]) * jnp.cos(x[1]) * jnp.exp(jnp.abs(1 - (jnp.sqrt(x[0]**2 + x[1]**2)/jnp.pi))))

def franke(x):
    '''range = (0,1)
    '''
    term1 = 0.75 * jnp.exp(-(9*x[0]-2)**2/4 - (9*x[1]-2)**2/4)
    term2 = 0.75 * jnp.exp(-(9*x[0]+1)**2/49 - (9*x[1]+1)/10)
    term3 = 0.5 * jnp.exp(-(9*x[0]-7)**2/4 - (9*x[1]-3)**2/4)
    term4 = -0.2 * jnp.exp(-(9*x[0]-4)**2 - (9*x[1]-7)**2)

    return term1 + term2 + term3 + term4
