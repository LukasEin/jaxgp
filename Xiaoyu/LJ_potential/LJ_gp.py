import jax.numpy as jnp
import numpy as onp

from jaxgp.kernels import RBF
from jaxgp.regression import ExactGPR

def mean_err(x):
    new = x[~onp.isnan(x)]
    return onp.mean(new)

def LJ_GP(ca, cb, saa, sab, sbb):
    def integrand_normal(ca, cb, saa, sab):
        term = 1. / (saa - sab*jnp.sqrt(ca/cb)) - 1
        return term / ca
    
    saa = jnp.array(saa)[1:-1]
    sab = jnp.array(sab)[1:-1]
    sbb = jnp.array(sbb)[1:-1]
    ca = jnp.array(ca)[1:-1]
    cb = jnp.array(cb)[1:-1]

    # models
    kernel = RBF()
    noise = jnp.ones((19,))*2.5e-3
    noise = jnp.hstack((jnp.zeros(1), noise))

    model_A = ExactGPR(kernel, noise=noise)
    model_B = ExactGPR(kernel, noise=noise)

    # train A
    X_train = (jnp.zeros((1,1)), ca.reshape(-1,1))
    Y_train = integrand_normal(ca, cb, saa, sab)
    Y_train = jnp.hstack((jnp.zeros(1), Y_train))

    model_A.train(X_train, Y_train)

    # train B
    X_train = (jnp.zeros((1,1)), cb.reshape(-1,1))
    Y_train = integrand_normal(cb, ca, sbb, sab)
    Y_train = jnp.hstack((jnp.zeros(1), Y_train))

    model_B.train(X_train, Y_train)

    # predict mu_A
    mean, std = model_A.eval(ca.reshape(-1,1))
    mu_a = mean.reshape(-1)
    mu_a_err = std.reshape(-1)

    # predict mu_B
    mean, std = model_B.eval(cb.reshape(-1,1))
    mu_b = mean.reshape(-1)
    mu_b_err = std.reshape(-1)

    return mu_a, mu_a_err, mu_b, mu_b_err