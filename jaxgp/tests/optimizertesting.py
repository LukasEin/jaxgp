from typing import Callable, Tuple, Union

import jax.numpy as jnp
from jax import Array, grad, jit, random, vmap

from .._src.regression.regression import ExactGPR, SparseGPR
from .._src.kernels import BaseKernel
from .._src.logger import Logger
   

def compare_optimizer_data(functions: list[Callable], ranges: list[Tuple[float, float]], names: list[str], optimizers: str, num_gridpoints: Array, in_dir: str, write: Callable):
    '''Compares optimizers over different functions in multiple ways:
    - how many true function values lie in the 68% and 95% confidence intervals
    - the maximum difference between the mean and the true function values
    - mean squared error over the whole grid
    - maximum value of the standard deviation

    Parameters
    ----------
    functions : list[Callable]
        list of functions for which the optimizers should be compared
    ranges : list[Tuple[float, float]]
        list of ranges over which the functions were evaluated
    names : list[str]
        names of the functions
    optimizers : str
        different optimizers that are to be compared
    num_gridpoints : Array
        shape (num_x1, num_x2), number of grid points along each axis
        Total number of gridpoints = Prod(num_gridpoints_i)
    in_dir : str
        directory in which the files to be compared are stored
    write : Callable
        function (str) -> Any, that somehow processes the information given in form of strings
    '''
    for fun, ran, name in zip(functions, ranges, names):
        _, Y = create_training_data_2D(0, num_gridpoints, ran, 0.0, fun)
        Y = Y[:,0]
        
        write("-"*70 + "\n")
        write(f"Current function: {name}\n")
        
        for optimizer in optimizers:
            means = jnp.load(f"{in_dir}/{name}means{optimizer}.npz")
            stds = jnp.load(f"{in_dir}/{name}stds{optimizer}.npz")

            write(f"optimizer: {optimizer}\n")

            for iter, (mean, std) in enumerate(zip(means.values(), stds.values())):
                conf68 = jnp.where(jnp.abs(Y-mean) <= std, 0, 1)
                conf95 = jnp.where(jnp.abs(Y-mean) <= 2*std, 0, 1)

                mse = jnp.mean((Y-mean)**2)

                maxdiff = jnp.max(jnp.abs(Y-mean))
                maxstd = jnp.max(std)

                write(f"iter {iter}: 68% = {jnp.mean(conf68):.5f}, 95% = {jnp.mean(conf95):.5f}, maxerr = {maxdiff:.5f}, mse = {mse:.5f}, maxstd = {maxstd:.5f}\n")

def create_optimizer_data(functions: list[Callable], ranges: list[Tuple[float, float]], names: list[str], optimizers: str, num_gridpoints: int, 
                          noise: Union[float, Array], seed: int, num_f_vals: int,  num_d_vals: int, kernel: BaseKernel, param_bounds: Tuple, 
                          param_shape: Tuple , iters_per_optimizer: int, in_dir: str) -> None:
    '''Calculates and saves the posterior means and stds as well as the convergens behavior of the kernel_parameter
    and stores everything in .npz files

    Parameters
    ----------
    functions : list[Callable]
        list of functions for which the optimizers should be compared
    ranges : list[Tuple[float, float]]
        list of ranges over which the functions were evaluated
    names : list[str]
        names of the functions
    optimizers : str
        different optimizers that are to be compared
    num_gridpoints : Array
        shape (num_x1, num_x2), number of grid points along each axis
        Total number of gridpoints = Prod(num_gridpoints_i)
    noise : Union[float, Array]
        noise parameter that was added to Y_train
        Array not yet implemented!
    seed : int
        seed for the random sampling
    num_f_vals : int
        number of function evaluations to put in random subset
    num_d_vals : int
        number of derivative evaluations to put in the random subset
        Both derivatives are sampled at the same subset of features
    kernel : BaseKernel
        kernel that describes the covariance between features
    param_bounds : Tuple
        tuple (min, max), range from which to randomly choose the initial kernel parameters
    param_shape : Tuple
        Tuple (n_params, ), shape of the initial kernel parameters
    iters_per_optimizer : int
        _description_
    in_dir : str
        directory in which the files to be compared are stored
    '''
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
            for elem in logger.iters_list:
                params.append(elem[0])
            jnp.savez(f"{in_dir}/{name}params{optimizer}_seed{seed}", *params)

def create_test_data_2D(X_train: Array, Y_train: Array, num_f_vals: int, num_d_vals: int, logger: Logger, 
                        kernel: BaseKernel, param_bounds: Tuple, param_shape: Tuple, noise: Union[float, Array], 
                        optimizer: str,  iters: int, evalgrid: Array, seed: int, *, sparse: bool=False, ref=0.1) -> Tuple[Array, Array]:
    '''Takes the full training set and creates a random subset to fit an exact gpr to.

    Parameters
    ----------
    X_train : Array
        shape (n_samples, n_features), full training features
    Y_train : Array
        shape (n_samples, 3), full training labels (function eval, d1 eval , d2 eval) for each training feature
    num_f_vals : int
        number of function evaluations to put in random subset
    num_d_vals : int
        number of derivative evaluations to put in the random subset
        Both derivatives are sampled at the same subset of features
    logger : Logger
        logs the optimization of the fitting parameters
    kernel : BaseKernel
        kernel that describes the covariance between features
    param_bounds : Tuple
        tuple (min, max), range from which to randomly choose the initial kernel parameters
    param_shape : Tuple
        Tuple (n_params, ), shape of the initial kernel parameters
    noise : Union[float, Array]
        noise parameter that was added to Y_train
        Array not yet implemented!
    optimizer : str
        type of optimizer to use in finding optimal kernel parameters
    iters : int
        number of restarts with the same training data (chooses new inital kernel parameters)
    evalgrid : Array
        grid on which to evaluate the posterior mean and std
    seed : int
        seed for the random sampling

    Returns
    -------
    Tuple[Array, Array]
        posterior means and stds evaluated on evalgrid
    '''
    key = random.PRNGKey(seed)
    means = []
    stds = []

    num_datapoints = X_train.shape[0]

    key, subkey = random.split(key)
    fun_perm = random.permutation(subkey, num_datapoints)[:num_f_vals]
    key, subkey = random.split(key)
    d_perm = random.permutation(subkey, num_datapoints)[:num_d_vals]

    X_fun = X_train[fun_perm]
    Y_fun = Y_train[fun_perm,0]
    X_d = X_train[d_perm]
    Y_d1 = Y_train[d_perm,1]
    Y_d2 = Y_train[d_perm,2]

    X = jnp.vstack((X_fun, X_d, X_d))
    Y = jnp.hstack((Y_fun, Y_d1, Y_d2))
    data_split = jnp.array([num_f_vals, num_d_vals, num_d_vals])

    if sparse:
        num_ref_points = int((num_d_vals + num_f_vals)*ref + 1)

        key, subkey = random.split(key)
        ref_perm = random.permutation(subkey, num_d_vals + num_f_vals)[:num_ref_points]
        X_ref = X[ref_perm]

    for i in range(iters):
        key, subkey = random.split(key)
        init_params = random.uniform(subkey, param_shape, minval=param_bounds[0], maxval=param_bounds[1])
        logger.log(f"# iter {i+1}: init params {init_params}")
        
        if sparse:
            model = SparseGPR(kernel, init_params, noise, X_ref, optimize_method=optimizer, logger=logger)
        else:
            model = ExactGPR(kernel, init_params, noise, optimize_method=optimizer, logger=logger)
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