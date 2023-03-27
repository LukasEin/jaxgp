import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import random
from ..kernels import RBF
from .GPR import ExactGPR

class Bootstrapper:
    def __init__(self, X_data, Y_data, data_split, X_predict, noise_var=1e-6, kernel=RBF(), batches=None, n_runs=1, seed=0):
        self.X_data = X_data
        self.Y_data = Y_data
        self.data_split = data_split
        self.X_predict = X_predict

        self.noise_var = noise_var

        self.kernel = kernel
        self.batches = batches
        self.n_runs = n_runs

        self.seed = seed

        self.permuted_X_data = None
        self.permuted_Y_data = None
        self.data_partials = None

        self.means = []
        self.stds = []

    def get_prediction(self):
        return self.means, self.stds

    def run(self):
        self._split_indices()

        model = ExactGPR(self.kernel, self.data_partials[0], self.data_partials[1:], noise_var=self.noise_var)

        key = random.PRNGKey(self.seed)

        for _ in range(self.n_runs):
            key, subkey = random.split(key)
            self._permute_data(subkey)

            self._epoch(model)

        # self.means /= (self.batches * self.n_runs)
        # self.stds /= (self.batches * self.n_runs)

    def _epoch(self, model):
        partial_X, partial_Y = self._split_data(0)

        model.fit(partial_X, partial_Y)
        means, stds = model.predict(self.X_predict,return_std=True)
        self.means.append(means)
        self.stds.append(stds)

        for i in range(1,self.batches):
            partial_X, partial_Y = self._split_data(i)

            model.fit(partial_X, partial_Y)
            # temp_means, temp_stds = model.predict(self.X_predict, return_std=True)
            means, stds = model.predict(self.X_predict, return_std=True)

            # means += temp_means
            # stds += temp_stds
            self.means.append(means)
            self.stds.append(stds)

        # self.means += means
        # self.stds += stds

    def _split_indices(self):
        self.data_partials = [elem // self.batches for elem in self.data_split]
    
    def _permute_data(self, key):
        permuted_indices = tuple(random.permutation(key, i) for i in self.data_split)

        permuted_X_data = self.X_data[permuted_indices[0]]
        permuted_Y_data = self.Y_data[permuted_indices[0]]

        sum_dims = self.data_split[0]
        for dim, indices in zip(self.data_split[1:], permuted_indices[1:]):
            temp = self.X_data[sum_dims:sum_dims+dim]
            permuted_X_data = jnp.concatenate((permuted_X_data,temp[indices]), axis=0)
            temp = self.Y_data[sum_dims:sum_dims+dim]
            permuted_Y_data = jnp.concatenate((permuted_Y_data,temp[indices]), axis=0)
            sum_dims += dim

        self.permuted_X_data, self.permuted_Y_data = permuted_X_data, permuted_Y_data

    def _split_data(self, index):
        partial_X = self.permuted_X_data[self.data_partials[0]*index:self.data_partials[0]*(index+1)]
        partial_Y = self.permuted_Y_data[self.data_partials[0]*index:self.data_partials[0]*(index+1)]

        sum_dims = self.data_split[0]
        for dim, partial in zip(self.data_split[1:], self.data_partials[1:]):
            temp = self.permuted_X_data[sum_dims+partial*index:sum_dims+partial*(index+1)]
            partial_X = jnp.concatenate((partial_X,temp), axis=0)
            temp = self.permuted_Y_data[sum_dims+partial*index:sum_dims+partial*(index+1)]
            partial_Y = jnp.concatenate((partial_Y,temp), axis=0)
            sum_dims += dim

        return partial_X, partial_Y