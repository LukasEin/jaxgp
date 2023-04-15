import jax.numpy as jnp


def sin2d(x):
    return jnp.sin(2*x[0] + x[1])

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
