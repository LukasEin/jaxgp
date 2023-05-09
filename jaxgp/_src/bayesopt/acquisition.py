import jax.numpy as jnp

from ..predict import full_predict

from jaxopt import ScipyBoundedMinimize

from jax import random

def upper_confidence_bound(cov_matrix, Y_data, X_split, kernel, params, bounds, eval_function):
    def minim_func(x):
        mean, std = full_predict(x, cov_matrix, Y_data, X_split, kernel, params)
        return -(mean + std)[0]
    
    key = random.PRNGKey(0)
    init_point = random.uniform(key, shape=bounds[0].shape, minval=bounds[0], maxval=bounds[1])
    
    solver = ScipyBoundedMinimize(fun=minim_func, method="L-BFGS-B")
    result = solver.run(init_point, bounds)

    X_next = result.params
    Y_next, isgrad = eval_function(X_next)

    return X_next, Y_next, isgrad

