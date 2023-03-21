import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve

class sparseGPR_ND:
    def __init__(self, kernel, n_datapoints=0, n_derivpoints=(0,), X_ref=None, *, sparse_method="ppa", noise_var=1e-6) -> None:
        '''
            supported kernels are found in the gaussion_process_regression.kernels folder

            num_datapoints (int)    ... number of function values included in the model
            num_derivpoints (tuple) ... tuple of numbers of derivatives included for each independent variable
        '''
        self.kernel = kernel

        self.n_datapoints = n_datapoints
        self.n_derivpoints = n_derivpoints
        self.n_referencepoints = len(X_ref)

        # X_ref must be given for now!!!!!
        if X_ref is None:
            raise ValueError("X_ref can't be None!")
        self.X_ref = X_ref
        self._K_ref = self.kernel.eval_func(X_ref,X_ref)

        sparse_methods = ["ppa", "sor", "ny"]
        if sparse_method not in sparse_methods:
            raise ValueError("Unrecognized sparsification method chosen!")
        self.sparse_method = sparse_method

        self.noise_var = noise_var
        self._fit_vector = None
        self._fit_matrix = None

        self.X_data = None
        self.Y_data = None

    def fit(self, X_data, Y_data) -> None:
        '''
            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        self.X_data = X_data
        self.Y_data = Y_data

        if self.sparse_method == "ppa" or self.sparse_method == "sor":
            self._fit_PPA_SoR()
        elif self.sparse_method == "ny":
            self._fit_nystrom()
        else:
            raise ValueError("Unrecognized sprasification method chosen")

    def predict(self,X, return_std=False):
        '''
            X.shape = (N, n_features)
        '''
        if self.sparse_method == "ppa":
            return self._predict_PPA(X,return_std)
        elif self.sparse_method == "sor":
            return self._predict_SoR(X,return_std)
        elif self.sparse_method == "ny":
            return self._predict_Ny(X,return_std)
        
        raise ValueError("Unrecognized sprasification method chosen")
    
    def _fit_PPA_SoR(self) -> None:
        '''
            Fits the GPR based on the Projected Process Approximation and Subset of Regressors
            The difference between the two methods lies in their predict function

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        K_MN = self.kernel.eval_func(self.X_ref,self.X_data[:self.n_datapoints])
        sum_dims = 0
        for dim in self.n_derivpoints:
            K_deriv = self.kernel.eval_dx2(self.X_ref,self.X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=0)
            sum_dims += dim
        K_MN = K_MN.T

        # added small positive diagonal to make the matrix positive definite
        self._fit_matrix = self.noise_var * self._K_ref + K_MN@K_MN.T + jnp.eye(len(self.X_ref))*1e-10
        self._fit_vector = K_MN@self.Y_data

    def _fit_nystrom(self) -> None:
        '''
            Fits the GPR based on the Nystroem Method

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        K_MN = self.kernel.eval_func(self.X_ref,self.X_data[:self.n_datapoints])
        sum_dims = 0
        for dim in self.n_derivpoints:
            K_deriv = self.kernel.eval_dx2(self.X_ref,self.X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=0)
            sum_dims += dim
        K_MN = K_MN.T

        # added small positive diagonal to make the matrix positive definite
        temp = self.noise_var * self._K_ref + K_MN@K_MN.T + jnp.eye(len(self.X_ref))*1e-10
        self._fit_matrix = (jnp.eye(len(self.X_data)) - K_MN.T@solve(temp,K_MN)) / self.noise_var

    def _predict_PPA(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Projected Process Approximation

            X.shape = (N, n_features)
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        ref_vectors = self.kernel.eval_func(self.X_ref,X)

        means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        if return_std:
            X_cov = self.kernel.eval_func(X,X)
            first_temp = jnp.array([k.T@solve(self._K_ref + jnp.eye(len(self.X_ref))*1e-10,k) for k in ref_vectors])
            second_temp =  self.noise_var * jnp.array([k.T@solve(self._fit_matrix,k) for k in ref_vectors])
            
            # stds = np.sqrt(X_cov + self.noise_var - first_temp + second_temp)
            stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
            
            return means, stds
        
        return means

    def _predict_SoR(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Subset of Regressors
            
            X.shape = (N, n_features)
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        ref_vectors = self.kernel.eval_func(self.X_ref,X)

        means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        if return_std:
            vars =  self.noise_var * jnp.array([k.T@solve(self._fit_matrix,k) for k in ref_vectors])
            stds = jnp.sqrt(vars)

            return means, stds
        
        return means
    
    def _predict_Ny(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Subset of Regressors
            
            X.shape = (N, n_features)
        '''
        if self._fit_matrix is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        full_vectors = self._create_covMatrix(X, self.X_data[:self.num_datapoints], self.kernel.eval)
        sum_dims = 0
        for dim in self.n_derivpoints:
            deriv_vectors = self._create_covMatrix(X, self.X_data[self.num_datapoints + sum_dims:self.n_datapoints + sum_dims + dim], self.kernel.dx2)
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=0)
            sum_dims += dim
        full_vectors = full_vectors.T

        means = full_vectors@self._fit_matrix@self.Y_data

        if return_std:
            X_cov = self.kernel.eval_func(X,X)
            temp = jnp.array([k.T@self._fit_matrix@k for k in full_vectors])
            stds = jnp.sqrt(X_cov - temp) # no noise term in the variance

            return means, stds

        return means