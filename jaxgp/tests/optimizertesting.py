from typing import Callable, Tuple, Union

import jax.numpy as jnp
from jax import Array, grad, jit, random, vmap

import jaxgp.regression as gpr
from jaxgp.kernels import BaseKernel
from jaxgp.utils import Logger


def create_optimizer_data(functions: list, ranges: list, names: list, optimizers: str, num_gridpoints: int, noise: float, seed: int, num_f_vals: int, 
                          num_d_vals: int, kernel: BaseKernel, param_bounds: Tuple, param_shape: Tuple , iters_per_optimizer: int, in_dir: str):
    for fun, ran, name in zip(functions, ranges, names):
        X_train, Y_train = create_training_data_2D(seed, num_gridpoints, ran, noise, fun)

        grid1 = jnp.linspace(*ran[0],100)
        grid2 = jnp.linspace(*ran[1],100)
        grid = jnp.array(jnp.meshgrid(grid1, grid2)).reshape(2,-1).T

        for optimizer in optimizers:
            print(f"Optimizer {optimizer}")
            logger = Logger(optimizer)

            means, stds = create_test_data_2D(X_train=X_train, Y_train=Y_train, num_f_vals=num_f_vals, num_d_vals=num_d_vals,
                                              logger=logger, kernel=kernel, param_bounds=param_bounds, param_shape=param_shape, noise=noise, optimizer=optimizer,iters=iters_per_optimizer, evalgrid=grid, seed=seed)
            
            jnp.savez(f"{in_dir}/{name}means{optimizer}", *means)
            jnp.savez(f"{in_dir}/{name}stds{optimizer}", *stds)
            params = []
            losses = []
            for elem in logger.iters_list:
                params.append(elem[0])
                losses.append(elem[1])
            jnp.savez(f"{in_dir}/{name}params{optimizer}", *params)
            jnp.savez(f"{in_dir}/{name}losses{optimizer}", *losses)   

def compare_optimizer_data(functions: list, ranges: list, names: list, optimizers: str, num_gridpoints: int, in_dir: str, write: Callable):
    for fun, ran, name in zip(functions, ranges, names,):
        X, Y = create_training_data_2D(0, num_gridpoints, ran, 0.0, fun)
        Y = Y[:,0]
        
        write("-"*70 + "\n")
        write(f"Current function: {name}\n")
        
        for optimizer in optimizers:
            means = jnp.load(f"{in_dir}/{name}means{optimizer}.npz")
            stds = jnp.load(f"{in_dir}/{name}std{optimizer}.npz")

            write(f"optimizer: {optimizer}\n")

            for iter, (mean, std) in enumerate(zip(means.values(), stds.values())):
                conf68 = jnp.where(jnp.abs(Y-mean) <= std, 0, 1)
                conf95 = jnp.where(jnp.abs(Y-mean) <= 2*std, 0, 1)

                mse = jnp.mean((Y-mean)**2)

                maxdiff = jnp.max(jnp.abs(Y-mean))
                maxstd = jnp.max(std)

                write(f"iter {iter}: 68% = {jnp.mean(conf68):.5f}, 95% = {jnp.mean(conf95):.5f}, maxerr = {maxdiff:.5f}, mse = {mse:.5f}, maxstd = {maxstd:.5f}\n")

def create_test_data_2D(X_train: Array, Y_train: Array, num_f_vals: list[int], num_d_vals: list[int], logger: Logger, kernel: BaseKernel, 
                   param_bounds: Tuple, param_shape: Tuple, noise: Union[float, Array], optimizer: str,  iters: int, evalgrid: Array, seed: int):
    key = random.PRNGKey(seed)
    means = []
    stds = []

    num_datapoints = X_train.shape[0]

    key, subkey = random.split(key)
    fun_perm = random.permutation(subkey, num_datapoints)[:num_f_vals]
    key, subkey = random.split(key)
    d1_perm = random.permutation(subkey, num_datapoints)[:num_d_vals]
    # key, subkey = random.split(key)
    # d2_perm = random.permutation(subkey, num_datapoints)[:num_d_vals]

    X_fun = X_train[fun_perm]
    Y_fun = Y_train[fun_perm,0]
    X_d1 = X_train[d1_perm]
    Y_d1 = Y_train[d1_perm,1]
    X_d2 = X_train[d1_perm]
    Y_d2 = Y_train[d1_perm,2]

    X = jnp.vstack((X_fun, X_d1, X_d2))
    Y = jnp.hstack((Y_fun, Y_d1, Y_d2))
    data_split = jnp.array([num_f_vals, num_d_vals, num_d_vals])

    for i in range(iters):
        key, subkey = random.split(key)
        init_params = random.uniform(subkey, param_shape, minval=param_bounds[0], maxval=param_bounds[1])
        logger.log(f"# iter {i+1}: init params {init_params}")

        model = gpr.ExactGPR(kernel, init_params, noise, optimize_method=optimizer, logger=logger)
        model.train(X, Y, data_split=data_split)
        m, s = model.eval(evalgrid)
        means.append(m)
        stds.append(s)

    return means, stds

def create_training_data_2D(seed: int, num_gridpoints: int, ranges: Tuple, noise: Union[float, Array], test_function: Callable) -> Tuple[Array, Array]:
    '''creates training data for 2D functions

    Parameters
    ----------
    seed : int
        seed for RNG noise creation
    num_gridpoints : int
        number of gridpoints in both dimensions, total returned datapoints are of size num_gridpoints**2
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

    X1 = jnp.linspace(*ranges[0],num_gridpoints[0])
    X2 = jnp.linspace(*ranges[1],num_gridpoints[1])
    X = jnp.array(jnp.meshgrid(X1, X2)).reshape(2,-1).T

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    Y = f(X) + noise*random.normal(subkey, (jnp.prod(num_gridpoints),))
    key, subkey = random.split(key)
    dY = df(X) + noise*random.normal(subkey, (jnp.prod(num_gridpoints),2))

    Y = jnp.hstack((Y.reshape(-1,1), dY))

    return X, Y