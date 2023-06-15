import jax.numpy as jnp
from jax import random, jit

from jaxgp.covar import full_covariance_matrix, sparse_covariance_matrix
from jaxgp.kernels import RBF

from timeit import timeit, repeat

def fun(x, noise=0.0, key = random.PRNGKey(0)):
    return (x[:,0]**2 + x[:,1] - 11)**2 / 800.0 + (x[:,0] + x[:,1]**2 -7)**2 / 800.0 + random.normal(key,(len(x),), dtype=jnp.float32)*noise

def grad(x, noise=0.0, key = random.PRNGKey(0)):
    dx1 = 4 * (x[:,0]**2 + x[:,1] - 11) * x[:,0] + 2 * (x[:,0] + x[:,1]**2 -7)
    dx2 = 2 * (x[:,0]**2 + x[:,1] - 11) + 4 * (x[:,0] + x[:,1]**2 -7) * x[:,1]

    return jnp.vstack((dx1, dx2)).T / 800.0 + random.normal(key,x.shape, dtype=jnp.float32)*noise

# Constants
BOUNDS = jnp.array([-5.0, 5.0])
NUM_F_VALS = 1
NOISE = 0.02
KERNEL = RBF()
KERNEL_PARAMS = jnp.ones(2)*jnp.log(2)

def _train_data(num_d_vals):
    # initial seed for the pseudo random key generation
    seed = 3

    # create new keys and randomly sample the above interval for training features
    key, subkey = random.split(random.PRNGKey(seed))
    x_func = random.uniform(subkey, (NUM_F_VALS, 2), minval=BOUNDS[0], maxval=BOUNDS[1])
    key, subkey = random.split(key)
    x_der = random.uniform(subkey, (num_d_vals,2), minval=BOUNDS[0], maxval=BOUNDS[1])

    X_split = [x_func,x_der]

    key, subkey = random.split(key)
    y_func = fun(x_func, NOISE, subkey)
    key, subkey = random.split(key)
    y_der = grad(x_der, NOISE, subkey)

    Y_train = jnp.hstack((y_func, y_der.reshape(-1)))

    return X_split, Y_train

def ref_from_data(X_split, num_ref_points):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    X_ref_rand = random.permutation(subkey, jnp.vstack(X_split))[:num_ref_points]

    return X_ref_rand

def full_timing(start, stop, step):
    times = []

    for i in range(start, stop, step):
        X_train, Y_train = _train_data(i)

        def test():
            X = jit(full_covariance_matrix)(X_train, Y_train, KERNEL, KERNEL_PARAMS, NOISE)


        times.append(repeat(test, number=10)[1:])
    times = jnp.array(times)
    avg_times = jnp.mean(times, axis=1)
    jnp.save(f"full_time_{start}_{stop}_{step}", avg_times)

def sparse_timing_fixed_percent(start, stop, step, percent):
    times = []

    for i in range(start, stop, step):
        X_train, Y_train = _train_data(i)
        num_ref_points = int(len(X_train[0]) + len(X_train[1])) + 1
        X_ref = ref_from_data(X_train, num_ref_points)

        def test():
            X = jit(sparse_covariance_matrix)(X_train, Y_train, X_ref, KERNEL, KERNEL_PARAMS, NOISE)


        times.append(repeat(test, number=10)[1:])
    times = jnp.array(times)
    avg_times = jnp.mean(times, axis=1)
    jnp.save(f"sparse_time_{start}_{stop}_{step}", avg_times)