from enum import Enum

from jax import jit
from jaxopt import ScipyBoundedMinimize


class OptimizerTypes(Enum):
    SLSQP = 0
    TNC = 1
    LBFGSB = 2

_optimizer_types = ("SLSQP", "TNC", "L-BFGS-B")

def parameter_optimize(fun, params, bounds, method: OptimizerTypes, callback=None, jit_fun=True, *args):
    if jit_fun:
        opt_fun = jit(fun)
    else:
        opt_fun = fun

    if callback is not None:
        callback(params)

    solver = ScipyBoundedMinimize(fun=opt_fun, method=_optimizer_types[method.value], callback=callback)
    result = solver.run(params, bounds, *args)

    print(result.state.success)

    return result.params