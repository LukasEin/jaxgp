import jax.numpy as jnp
from jax.scipy.linalg import solve
from ..kernels import RBF
from jax import jit, vmap
from functools import partial
import time

class GaussianProcessRegressor:
    def __init__(self, kernel=RBF(), n_datapoints=0, n_derivpoints=(0,), X_ref=jnp.zeros(0), *, sparse_method="full", noise_var=1e-6) -> None:
        '''
            supported kernels are found in the gaussion_process_regression.kernels folder

            num_datapoints (int)    ... number of function values included in the model
            num_derivpoints (tuple) ... tuple of numbers of derivatives included for each independent variable
        '''
        self.kernel = kernel

        self.n_datapoints = n_datapoints
        self.n_derivpoints = n_derivpoints

        if sparse_method != "full":
            self.n_referencepoints = len(X_ref)

            # X_ref must be given for now!!!!!
            if X_ref is None:
                raise ValueError("X_ref can't be None!")
            self.X_ref = X_ref
            # print("Start building K_ref.")
            # start = time.time()
            self._K_ref = self._build_cov_func(X_ref, X_ref)
            # print(f"Took {time.time() - start}\nFinished building K_ref")

            sparse_methods = ["ppa", "sor", "ny"]
            if sparse_method not in sparse_methods:
                raise ValueError("Unrecognized sparsification method chosen!")
        
        self.sparse_method = sparse_method

        self.noise_var = noise_var
        self._fit_vector = None
        self._fit_matrix = None

        self.X_data = None
        self.Y_data = None

        self.diag_add = 1e-6

        self.full_vectors = None

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
            self._fit_Ny()
        elif self.sparse_method == "full":
            self._fit_Full()
        else:
            raise ValueError("Unrecognized sprasification method chosen")

    def predict(self,X, return_std=False):
        '''
            X.shape = (N, n_features)
        '''
        if self.sparse_method == "ppa":
            return self._predict_PPA(X, return_std)
        elif self.sparse_method == "sor":
            return self._predict_SoR(X, return_std)
        elif self.sparse_method == "ny":
            return self._predict_Ny(X, return_std)
        elif self.sparse_method == "full":
            return self._predict_Full(X, return_std)
        
        raise ValueError("Unrecognized sprasification method chosen")
    
    def _fit_PPA_SoR(self) -> None:
        '''
            Fits the GPR based on the Projected Process Approximation and Subset of Regressors
            The difference between the two methods lies in their predict function

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        K_MN = self._build_cov_func(self.X_ref, self.X_data[:self.n_datapoints])
        sum_dims = 0
        for i,dim in enumerate(self.n_derivpoints):
            K_deriv = self._build_cov_dx2(
                    self.X_ref, 
                    self.X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim],
                    index=i
            )
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)
            sum_dims += dim

        # added small positive diagonal to make the matrix positive definite
        self._fit_matrix = self.noise_var * self._K_ref + K_MN@K_MN.T + jnp.eye(len(self.X_ref)) * self.diag_add
        self._fit_vector = K_MN@self.Y_data

    def _fit_Full(self) -> None:
        '''
            Fits the full GPR Model

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        
        K_NN = self._build_cov_func(self.X_data[:self.n_datapoints],self.X_data[:self.n_datapoints])
        sum_dims = 0
        for i,dim in enumerate(self.n_derivpoints):
            K_mix = self._build_cov_dx2(
                self.X_data[:self.n_datapoints],
                self.X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim],
                index=i
            )
            K_NN = jnp.concatenate((K_NN,K_mix),axis=1)
            sum_dims += dim
        
        sum_dims_1 = 0
        sum_dims_2 = 0
        for i,dim_1 in enumerate(self.n_derivpoints):
            K_mix = self._build_cov_dx2(
                    self.X_data[:self.n_datapoints],
                    self.X_data[self.n_datapoints+sum_dims_1:self.n_datapoints+sum_dims_1+dim_1],
                    index=i
                )
            for j,dim_2 in enumerate(self.n_derivpoints):
                K_derivs = self._build_cov_ddx1x2(
                        self.X_data[self.n_datapoints+sum_dims_1:self.n_datapoints+sum_dims_1+dim_1],
                        self.X_data[self.n_datapoints+sum_dims_2:self.n_datapoints+sum_dims_2+dim_2],
                        index_1=i, index_2=j
                    )
                K_mix = jnp.concatenate((K_mix,K_derivs),axis=0)
                sum_dims_2 += dim_2
            K_NN = jnp.concatenate((K_NN,K_mix.T),axis=0)
            sum_dims_1 += dim_1
            sum_dims_2 = 0

        self._fit_matrix = jnp.eye(len(self.X_data)) * (self.noise_var + self.diag_add) + K_NN
    
    def _fit_Ny(self) -> None:
        '''
            Fits the GPR based on the Nystroem Method

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        K_MN = self._build_cov_func(self.X_ref, self.X_data[:self.n_datapoints])
        sum_dims = 0
        for i,dim in enumerate(self.n_derivpoints):
            K_deriv = self._build_cov_dx2(
                    self.X_ref, 
                    self.X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim],
                    index=i
                )
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=0)
            sum_dims += dim
        K_MN = K_MN.T

        # added small positive diagonal to make the matrix positive definite
        temp = self.noise_var * self._K_ref + K_MN@K_MN.T + jnp.eye(len(self.X_ref)) * self.diag_add
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
        
        ref_vectors = self._build_cov_func(self.X_ref, X).T

        means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        if return_std:
            X_cov = self._build_cov_vector(X)
            # X_cov = jnp.array([self.kernel.eval_func(x, x) for x in X])

            first_temp = self._build_xTAx(self._K_ref + jnp.eye(len(self.X_ref)) * self.diag_add, ref_vectors)
            second_temp = self.noise_var * self._build_xTAx(self._fit_matrix, ref_vectors)
            # first_temp = jnp.array([k.T@solve(self._K_ref + jnp.eye(len(self.X_ref)) * self.diag_add,k) for k in ref_vectors])
            # second_temp =  self.noise_var * jnp.array([k.T@solve(self._fit_matrix,k) for k in ref_vectors])
            
            stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
            
            return means, stds
        
        return means

    def _predict_SoR(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Subset of Regressors Aprroximation
            
            X.shape = (N, n_features)
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        ref_vectors = self._build_cov_func(self.X_ref, X).T

        means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        if return_std:
            vars = self.noise_var * self._build_xTAx(self._fit_matrix, ref_vectors)
            # vars = self.noise_var * jnp.array([k.T@solve(self._fit_matrix,k) for k in ref_vectors])
            stds = jnp.sqrt(vars)

            return means, stds
        
        return means
    
    def _predict_Ny(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Subset of Regressors
            
            X.shape = (N, n_features)
        '''
        if self._fit_matrix is None or self.Y_data is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        full_vectors = self._build_cov_func(X, self.X_data[:self.n_datapoints])
        sum_dims = 0
        for i,dim in enumerate(self.n_derivpoints):
            deriv_vectors = self._build_cov_dx2(
                    X, 
                    self.X_data[self.n_datapoints + sum_dims:self.n_datapoints + sum_dims + dim],
                    index=i
                )
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=0)
            sum_dims += dim
        full_vectors = full_vectors.T

        means = full_vectors@self._fit_matrix@self.Y_data

        if return_std:
            X_cov = self._build_cov_vector(X)
            # X_cov = jnp.array([self.kernel.eval_func(x, x) for x in X])
            temp = jnp.array([k.T@self._fit_matrix@k for k in full_vectors])       
            stds = jnp.sqrt(X_cov - temp) # no noise term in the variance

            return means, stds

        return means

    def _predict_Full(self, X, return_std=False):
        if self._fit_matrix is None or self.Y_data is None:
            raise ValueError("GPR not correctly fitted!")
        
        full_vectors = self._build_cov_func(X, self.X_data[:self.n_datapoints])
        # full_vectors = self.kernel.eval_func(X, self.X_data[:self.n_datapoints])
        sum_dims = 0
        for i,dim in enumerate(self.n_derivpoints):
            deriv_vectors = self._build_cov_dx2(
                    X, 
                    self.X_data[self.n_datapoints + sum_dims:self.n_datapoints + sum_dims + dim],
                    index=i
                )
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)
            sum_dims += dim
        # full_vectors = full_vectors.T

        # self.full_vectors = full_vectors

        means = full_vectors@solve(self._fit_matrix,self.Y_data)

        if return_std:
            X_cov = jnp.array([self.kernel.eval_func(x, x) for x in X])
            temp = jnp.array([k.T@solve(self._fit_matrix,k) for k in full_vectors])          
            stds = jnp.sqrt(X_cov - temp) # no noise term in the variance

            return means, stds

        return means
    
    @partial(vmap, in_axes=(None, None, 0))
    def _build_xTAx(self, A, X):
        '''
            X.shape = (N,M)
            A.shape = (M,M)

            output.shape = (N,)

            Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
        '''
        return X.T@solve(A,X)
    
    @partial(vmap, in_axes=(None,0))
    def _build_cov_vector(self, X):
        '''
            X1.shape = (N, n_features)

            output.shape = (N,)

            Builds a vector of the covariance of all X[:,i] with them selves.
        '''
        return self.kernel.eval_func(X, X)

    # @jit
    @partial(vmap, in_axes=(None,0,None))
    @partial(vmap, in_axes=(None,None,0))
    def _build_cov_func(self, X1, X2):
        '''
            X1.shape = (N1, n_features)
            X2.shape = (N2, n_features)

            output.shape = (N1, N2)

            Builds the covariance matrix between the elements of X1 and X2
            based on inputs representing values of the target function.

        '''
        return self.kernel.eval_func(X1, X2)
        # return jnp.array([jnp.apply_along_axis(self.kernel.eval_func, 1, X1, x) for x in X2])
    
    # @jit
    def _build_cov_dx2(self, X1, X2, index):
        '''
            X1.shape = (N1, n_features)
            X2.shape = (N2, n_features)
            
            index in range(0, n_features - 1), derivative of the kernel
                is taken w.r.t. to X_2[:,index].

            output.shape = (N1, N2)

            Builds the covariance matrix between the elements of X1 and X2
            based on X1 representing values of the target function and X2
            representing derivative values of the target function.
        '''
        func = lambda A, B: self.kernel.eval_dx2(A,B, index) 
        func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
        return func(X1, X2)
        # return jnp.array([jnp.apply_along_axis(self.kernel.eval_dx2, 1, X1, x, index) for x in X2])
    
    # @jit
    def _build_cov_ddx1x2(self, X1, X2, index_1, index_2):
        '''
            X1.shape = (N1, n_features)
            X2.shape = (N2, n_features)
            
            index_i in range(0, n_features - 1), double derivative of the 
                kernel is taken w.r.t. to X1[:,index_1] and X2[:,index_2].

            output.shape = (N1, N2)

            Builds the covariance matrix between the elements of X1 and X2
            based on X1 and X2 representing derivative values of the target 
            function.
        '''
        func = lambda A, B: self.kernel.eval_ddx1x2(A, B, index_1, index_2) 
        func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
        return func(X1, X2)
        # return jnp.array([jnp.apply_along_axis(self.kernel.eval_ddx1x2,1, X1, x, index_1, index_2) for x in X2])