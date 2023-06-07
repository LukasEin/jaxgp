from jaxopt import ScipyBoundedMinimize
from jax import jit

optimizers = ("L-BFGS-B", "TNC", "SLSQP")

def optimize(fun, params, bounds, method, callback=None, jit_fun=True, *args):
    if jit_fun:
        opt_fun = jit(fun)
    else:
        opt_fun = fun

    solver = ScipyBoundedMinimize(fun=opt_fun, method=optimizers[method], callback=callback)
    result = solver.run(params, bounds, *args)

    print(result.state.success)
    if callback is not None:
        callback.write()

    return result.params