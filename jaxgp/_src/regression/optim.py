from enum import Enum

from jax import jit
from jaxopt import ScipyBoundedMinimize


class Optimizer(Enum):
    SLSQP = 0
    TNC = 1
    LBFGSB = 2

optimizers = ("SLSQP", "TNC", "L-BFGS-B")

def optimize(fun, params, bounds, method: Optimizer, callback=None, jit_fun=True, *args):
    if jit_fun:
        opt_fun = jit(fun)
    else:
        opt_fun = fun

    if callback is not None:
        callback(params)

    solver = ScipyBoundedMinimize(fun=opt_fun, method=optimizers[method.value], callback=callback)
    result = solver.run(params, bounds, *args)

    print(result.state.success)

    return result.params