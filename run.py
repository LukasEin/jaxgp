import jax.numpy as jnp
from jaxgp.tests import testfunctions, optimizertesting

from jaxgp.utils import Logger
from jaxgp.kernels import RBF

def main():
    optimizers = ["L-BFGS-B"]#, "TNC", "SLSQP"]
    fun = lambda x: testfunctions.himmelblau(x)/800.0
    num_gridpoints = jnp.array([100,100])
    ran = (jnp.array([-5.0,5.0]), jnp.array([-5.0,5.0]))

    in_dir = "./data_files/different_number_of_datapoints"

    noise = 0.1
    seed = 0

    grid1 = jnp.linspace(*ran[0],100)
    grid2 = jnp.linspace(*ran[1],100)
    grid = jnp.array(jnp.meshgrid(grid1, grid2)).reshape(2,-1).T

    f_vals = [20,]#[20, 50]
    d_vals = [800,]#[5, 20, 50, 100, 200, 400, 800]

    kernel = RBF(3)
    param_shape = (3,)
    param_bounds = (1e-3, 10.0)

    iters_per_optimizer = 5

    X_train, Y_train = optimizertesting.create_training_data_2D(seed, num_gridpoints, ran, noise, fun)

    for num_f_vals in f_vals:
        print(f"Number function evals: {num_f_vals}")
        for num_d_vals in d_vals:        
            print(f"Number derivative evals: {num_d_vals}")
            for optimizer in optimizers:
                print(f"Optimizer: {optimizer}")
                logger = Logger(optimizer)

                means, stds = optimizertesting.create_test_data_2D(X_train=X_train, Y_train=Y_train, num_f_vals=num_f_vals, num_d_vals=num_d_vals,
                                                    logger=logger, kernel=kernel, param_bounds=param_bounds, param_shape=param_shape, noise=noise, optimizer=optimizer,iters=iters_per_optimizer, evalgrid=grid, seed=seed)
                
                jnp.savez(f"{in_dir}/him_f{num_f_vals}d{num_d_vals}means{optimizer}", *means)
                jnp.savez(f"{in_dir}/him_f{num_f_vals}d{num_d_vals}stds{optimizer}", *stds)
                params = []
                losses = []
                for elem in logger.iters_list:
                    params.append(elem[0])
                    losses.append(elem[1])
                jnp.savez(f"{in_dir}/him_f{num_f_vals}d{num_d_vals}params{optimizer}", *params)
                jnp.savez(f"{in_dir}/him_f{num_f_vals}d{num_d_vals}losses{optimizer}", *losses)

if __name__ == "__main__":
    main()
