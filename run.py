import jax.numpy as jnp
from jaxgp.tests import testfunctions, optimizertesting

from jaxgp.utils import Logger
from jaxgp.kernels import RBF, Periodic

def main():
    optimizers = ["L-BFGS-B", "TNC", "SLSQP"]
    fun = testfunctions.sin2d
    num_gridpoints = jnp.array([100,100])

    in_dir = "./data_files/different_number_of_datapoints/extended/periodic"

    noise = 0.1
    seed = 0

    f_vals = [1, 5, 20, 50]
    d_vals = [5, 20, 50, 100, 200, 400, 800]

    # kernel = RBF(3)
    kernel = Periodic()
    param_shape = (3,)
    param_bounds = (1e-3, 10.0)

    iters_per_optimizer = 5

    train_ran = (jnp.array([0.0,jnp.pi]), jnp.array([0.0,jnp.pi]))
    X_train, Y_train = optimizertesting.create_training_data_2D(seed, num_gridpoints, train_ran, noise, fun)

    test_rans = [(jnp.array([0.0,1.5*jnp.pi]), jnp.array([0.0,1.5*jnp.pi])),
                 (jnp.array([0.0,2*jnp.pi]), jnp.array([0.0,2*jnp.pi]))]
    names = ["1.5", "2.0"]

    for name,test_ran in zip(names,test_rans):
        print("-"*80)
        print(f"Extended interval: {name}pi")
        grid1 = jnp.linspace(*test_ran[0],100)
        grid2 = jnp.linspace(*test_ran[1],100)
        grid = jnp.array(jnp.meshgrid(grid1, grid2)).reshape(2,-1).T
        for num_f_vals in f_vals:
            print("-"*80)
            print(f"Number function evals: {num_f_vals}")
            for num_d_vals in d_vals:        
                print("-"*80)    
                print(f"Number derivative evals: {num_d_vals}")
                for optimizer in optimizers:
                    print("-"*80)
                    print(f"Optimizer: {optimizer}")
                    logger = Logger(optimizer)

                    means, stds = optimizertesting.create_test_data_2D(X_train=X_train, Y_train=Y_train, num_f_vals=num_f_vals, num_d_vals=num_d_vals,
                                                        logger=logger, kernel=kernel, param_bounds=param_bounds, param_shape=param_shape, noise=noise, optimizer=optimizer,iters=iters_per_optimizer, evalgrid=grid, seed=seed)
                    
                    jnp.savez(f"{in_dir}/sin_{name}pi_f{num_f_vals}d{num_d_vals}means{optimizer}", *means)
                    jnp.savez(f"{in_dir}/sin_{name}pi_f{num_f_vals}d{num_d_vals}stds{optimizer}", *stds)
                    params = []
                    losses = []
                    for elem in logger.iters_list:
                        params.append(elem[0])
                        losses.append(elem[1])
                    jnp.savez(f"{in_dir}/sin_{name}pi_f{num_f_vals}d{num_d_vals}params{optimizer}", *params)
                    jnp.savez(f"{in_dir}/sin_{name}pi_f{num_f_vals}d{num_d_vals}losses{optimizer}", *losses)

if __name__ == "__main__":
    main()
