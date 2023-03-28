import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.optimize import minimize
from ..kernels import BaseKernel, RBF
from jax import jit, vmap
from functools import partial
from typing import Iterable, Tuple

class BaseGPR:
    def __init__(self, kernel: BaseKernel = RBF(), data_split: Iterable = (0, 0), init_kernel_params: Iterable = (1.0, 1.0), noise_var: float = 1e-6) -> None:
        '''
            supported kernels are found in gaussion_process.kernels

            data_split : Iterable       ... first element:      number of (noisy) function evaluations
                                            further elements:   number of (noisy) derivative evaluations
            kernel_params : Iterable    ... initial parameters for the MLE parameter optimization
            noise_var : float           ... assumed noise in the evaluation of data
        '''
        self.kernel = kernel
        self.params = jnp.array((noise_var,) + init_kernel_params)
        self.data_split = jnp.array(data_split)

        # initialized variables to save from the fitting step
        self._fit_vector = None
        self._fit_matrix = None
        self.X_data = None
            
    def fit(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray) -> None:
        '''
            Fits the GPR Model to the given data

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)

            Thin wrapper with all side effects around the pure functions 
            self.optimize and self.forward that do the actual work.
        '''
        self.X_data = X_data
        self.params = self.optimize(self.params, X_data, Y_data, self.data_split)
        self._fit_matrix, self._fit_vector = self.forward(X_data, Y_data, self.params, self.data_split)
    
    def forward(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray, params: Iterable, data_split: Iterable) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        raise NotImplementedError("Forward function was not implemented in derived GP Regressor.")

    def predict(self, X: jnp.DeviceArray, return_std: bool = False):
        '''
            Predicts the posterior mean (and std if return_std=True) at each point in X
            X.shape = (N, n_features)
            
            Thin wrapper with all side effects around the pure function 
            self.eval that does the actual work.
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        
        return self.eval(X, self.X_data, self.data_split, self._fit_matrix, self._fit_vector, self.params, return_std)
    
    def eval(self, X: jnp.DeviceArray, X_data: jnp.DeviceArray, data_split: jnp.DeviceArray, fitmatrix: jnp.DeviceArray, 
             fitvector: jnp.DeviceArray, params: jnp.DeviceArray, return_std: bool = False) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        raise NotImplementedError("Forward function was not implemented in derived GP Regressor.")
    
    def negativeLogLikelyhoodEstimate(self, params: jnp.DeviceArray, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray, data_split) -> jnp.float32:
        fitmatrix, fitvector = self.forward(X_data, Y_data, params, data_split)
        _, logdet = jnp.linalg.slogdet(fitmatrix)
        return -0.5*(len(fitvector) * jnp.log(2*jnp.pi) + logdet + fitvector.T@solve(fitmatrix,fitvector))

    def optimize(self, initial_params: jnp.DeviceArray, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray, data_split) -> jnp.DeviceArray:
        # result = jit(minimize(self.negativeLogLikelyhoodEstimate, initial_params, (X_data, Y_data), method="BFGS"))
        result = minimize(self.negativeLogLikelyhoodEstimate, initial_params, (X_data, Y_data, data_split), method="BFGS")

        if result.success:
            if result.status == 1:
                print("Max iterations reached!")
                return result.x
            if result.status == 0:
                return result.x
            
        raise ValueError("An error occured while maximizing the log likelyhood")

    # @partial(jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, None, 0))
    def _build_xTAx(self, A: jnp.DeviceArray, X:jnp.DeviceArray) -> jnp.DeviceArray:
        '''
            X.shape = (N,M)
            A.shape = (M,M)

            output.shape = (N,)

            Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
        '''
        return X.T@solve(A,X)
    
    # @partial(jit, static_argnums=(0,))
    # @partial(vmap, in_axes=(None,0))
    def _build_cov_vector(self, X: jnp.DeviceArray, params: Iterable) -> jnp.DeviceArray:
        '''
            X1.shape = (N, n_features)

            output.shape = (N,)

            Builds a vector of the covariance of all X[:,i] with them selves.
        '''
        func = lambda A: self.kernel.eval_func(A, A, params)
        func = vmap(func, in_axes=(0))
        return func(X)

    # @partial(jit, static_argnums=(0,))
    def _CovMatrix_Kernel(self, X1: jnp.DeviceArray, X2: jnp.DeviceArray, params: Iterable) -> jnp.DeviceArray:
        '''
            X1.shape = (N1, n_features)
            X2.shape = (N2, n_features)

            output.shape = (N1, N2)

            Builds the covariance matrix between the elements of X1 and X2
            based on inputs representing values of the target function.
        '''
        func = lambda A, B: self.kernel.eval_func(A, B, params)
        func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
        return func(X1, X2)
    
    # @partial(jit, static_argnums=(0,))
    def _CovMatrix_KernelGrad(self, X1: jnp.DeviceArray, X2: jnp.DeviceArray, index: int, params: Iterable) -> jnp.DeviceArray:
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
        func = lambda A, B: self.kernel.eval_dx2(A,B, index, params) 
        func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
        return func(X1, X2)
    
    # @partial(jit, static_argnums=(0,))
    def _CovMatrix_KernelHess(self, X1: jnp.DeviceArray, X2: jnp.DeviceArray, index_1: int, index_2: int, params: Iterable) -> jnp.DeviceArray:
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
        func = lambda A, B: self.kernel.eval_ddx1x2(A, B, index_1, index_2, params) 
        func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
        return func(X1, X2)

class ExactGPR(BaseGPR):
    def __init__(self, kernel: BaseKernel = RBF(), data_split: Iterable = (0, 0), kernel_params: Iterable = (1.0, 1.0), noise_var: float = 1e-6) -> None:
        super().__init__(kernel, data_split, kernel_params, noise_var)

    # @partial(jit, static_argnums=(0,4))
    def forward(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray, params: Iterable, data_split: Iterable) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        # Build the full covariance Matrix between all datapoints in X_data depending on if they   
        # represent function evaluations or derivative evaluations
        sum_splits = [jnp.sum(data_split[:i+1]) for i,_ in enumerate(data_split[1:])]
        X_split = jnp.split(X_data, sum_splits)

        K_NN = self._CovMatrix_Kernel(X_split[0], X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_mix = self._CovMatrix_KernelGrad(X_split[0], elem, index=i, params=params[1:])
            K_NN = jnp.concatenate((K_NN,K_mix),axis=1)
        
        for i,elem_1 in enumerate(X_split[1:]):
            K_mix = self._CovMatrix_KernelGrad(X_split[0], elem_1, index=i, params=params[1:])

            for j,elem_2 in enumerate(X_split[1:]):
                K_derivs = self._CovMatrix_KernelHess(elem_1, elem_2, index_1=i, index_2=j, params=params[1:])
                K_mix = jnp.concatenate((K_mix,K_derivs),axis=0)

            K_NN = jnp.concatenate((K_NN,K_mix.T),axis=0)

        # additional small diagonal element added for 
        # numerical stability of the inversion and determinant
        return (jnp.eye(len(X_data)) * (params[0] + 1e-6) + K_NN), (Y_data)
    
    # @partial(jit, static_argnums=(0,6))
    def eval(self, X: jnp.DeviceArray, X_data: jnp.DeviceArray, data_split: jnp.DeviceArray, 
             fitmatrix: jnp.DeviceArray, fitvector: jnp.DeviceArray, params, return_std: bool = False) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        sum_splits = [jnp.sum(data_split[:i+1]) for i,_ in enumerate(data_split[1:])]
        X_split = jnp.split(X_data, sum_splits)
        
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_KernelGrad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector)

        if return_std:
            X_cov = self._build_cov_vector(X, params[1:])  
            temp = self._build_xTAx(fitmatrix, full_vectors)      
            stds = jnp.sqrt(X_cov - temp) # no noise term in the variance

            return means, stds

        return means

class ApproximateGPR(BaseGPR):
    def __init__(self, kernel =RBF(), n_datapoints: int = 0, n_derivpoints: Iterable = (0, ), X_ref: jnp.DeviceArray=None, noise_var: float = 0.000001) -> None:
        '''
            Approximates the full GP regressor by the Projected Process Approximation.
        '''
        super().__init__(kernel, n_datapoints, n_derivpoints, noise_var)

        if X_ref is None:
            raise ValueError("X_ref can't be None!")
        self.X_ref = X_ref
        self._K_ref = self._build_cov_func(X_ref, X_ref)

    # @partial(jit, static_argnums=(0,))
    def forward(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray):
        K_MN = self._build_cov_func(self.X_ref, X_data[:self.n_datapoints])
        sum_dims = 0
        for i,dim in enumerate(self.n_derivpoints):
            K_deriv = self._build_cov_dx2(
                    self.X_ref, 
                    X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim],
                    index=i
            )
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)
            sum_dims += dim
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = self.noise_var * self._K_ref + K_MN@K_MN.T + jnp.eye(len(self.X_ref)) * self.diag_add
        fit_vector = K_MN@self.Y_data
        
        return fit_matrix, fit_vector

    def fit(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray):
        '''
            Fits the GPR based on the Projected Process Approximation

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)
        '''
        self.X_data = X_data
        self.Y_data = Y_data

        self._fit_matrix, self._fit_vector = self.forward(X_data, Y_data)

    # @partial(jit, static_argnums=(0,))
    def predict(self, X: jnp.DeviceArray, return_std: bool = False):
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

            first_temp = self._build_xTAx(self._K_ref + jnp.eye(len(self.X_ref)) * self.diag_add, ref_vectors)
            second_temp = self.noise_var * self._build_xTAx(self._fit_matrix, ref_vectors)
            
            stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
            
            return means, stds
        
        return means

# class BaseGPR:
#     def __init__(self, kernel = RBF(), n_datapoints: int = 0, n_derivpoints: Iterable = (0, ), noise_var: float = 0.000001) -> None:
#         '''
#             supported kernels are found in gaussion_process.kernels

#             n_datapoints : int          ... number of (noisy) function evaluations
#             n_derivpoints : Iterable    ... number of (noisy) derivative evaluations
#             noise_var : float           ... assumed noise in the evaluation of data
#         '''
#         self.kernel = kernel
#         self.n_datapoints = n_datapoints
#         self.n_derivpoints = n_derivpoints
#         self.noise_var = noise_var

#         self._fit_vector = None
#         self._fit_matrix = None

#         self.X_data = None
#         self.Y_data = None

#         # additional small diagonal element added to all matrices 
#         # that are going to be inverted for numerical stability
#         self.diag_add = 1e-6
            
#     def fit(self, X_data, Y_data):
#         raise NotImplementedError("Fit function was not implemented in derived GP Regressor.")

#     def predict(self, X):
#         raise NotImplementedError("Predict function was not implemented in derived GP Regressor.")

#     @partial(jit, static_argnums=(0,))
#     @partial(vmap, in_axes=(None, None, 0))
#     def _build_xTAx(self, A: jnp.DeviceArray, X:jnp.DeviceArray):
#         '''
#             X.shape = (N,M)
#             A.shape = (M,M)

#             output.shape = (N,)

#             Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
#         '''
#         return X.T@solve(A,X)
    
#     @partial(jit, static_argnums=(0,))
#     @partial(vmap, in_axes=(None,0))
#     def _build_cov_vector(self, X: jnp.DeviceArray):
#         '''
#             X1.shape = (N, n_features)

#             output.shape = (N,)

#             Builds a vector of the covariance of all X[:,i] with them selves.
#         '''
#         return self.kernel.eval_func(X, X)

#     @partial(jit, static_argnums=(0,))
#     @partial(vmap, in_axes=(None,0,None))
#     @partial(vmap, in_axes=(None,None,0))
#     def _build_cov_func(self, X1: jnp.DeviceArray, X2: jnp.DeviceArray):
#         '''
#             X1.shape = (N1, n_features)
#             X2.shape = (N2, n_features)

#             output.shape = (N1, N2)

#             Builds the covariance matrix between the elements of X1 and X2
#             based on inputs representing values of the target function.
#         '''
#         return self.kernel.eval_func(X1, X2)
#         # return jnp.array([jnp.apply_along_axis(self.kernel.eval_func, 1, X1, x) for x in X2])
    
#     @partial(jit, static_argnums=(0,))
#     def _build_cov_dx2(self, X1: jnp.DeviceArray, X2: jnp.DeviceArray, index: int):
#         '''
#             X1.shape = (N1, n_features)
#             X2.shape = (N2, n_features)
            
#             index in range(0, n_features - 1), derivative of the kernel
#                 is taken w.r.t. to X_2[:,index].

#             output.shape = (N1, N2)

#             Builds the covariance matrix between the elements of X1 and X2
#             based on X1 representing values of the target function and X2
#             representing derivative values of the target function.
#         '''
#         func = lambda A, B: self.kernel.eval_dx2(A,B, index) 
#         func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
#         return func(X1, X2)
#         # return jnp.array([jnp.apply_along_axis(self.kernel.eval_dx2, 1, X1, x, index) for x in X2])
    
#     @partial(jit, static_argnums=(0,))
#     def _build_cov_ddx1x2(self, X1: jnp.DeviceArray, X2: jnp.DeviceArray, index_1: int, index_2: int):
#         '''
#             X1.shape = (N1, n_features)
#             X2.shape = (N2, n_features)
            
#             index_i in range(0, n_features - 1), double derivative of the 
#                 kernel is taken w.r.t. to X1[:,index_1] and X2[:,index_2].

#             output.shape = (N1, N2)

#             Builds the covariance matrix between the elements of X1 and X2
#             based on X1 and X2 representing derivative values of the target 
#             function.
#         '''
#         func = lambda A, B: self.kernel.eval_ddx1x2(A, B, index_1, index_2) 
#         func = vmap(vmap(func, in_axes=(None,0)), in_axes=(0,None))
#         return func(X1, X2)
#         # return jnp.array([jnp.apply_along_axis(self.kernel.eval_ddx1x2,1, X1, x, index_1, index_2) for x in X2])

# class ExactGPR(BaseGPR):
#     def __init__(self, kernel = RBF(), n_datapoints: int = 0, n_derivpoints: Iterable = (0, ), noise_var: float = 0.000001) -> None:
#         super().__init__(kernel, n_datapoints, n_derivpoints, noise_var)

#     # @partial(jit, static_argnums=(0,))
#     def forward(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray):
#         # Build the full covariance Matrix between all datapoints in X_data depending on if they   
#         # represent function evaluations or derivative evaluations
#         K_NN = self._build_cov_func(X_data[:self.n_datapoints],X_data[:self.n_datapoints])
#         sum_dims = 0
#         for i,dim in enumerate(self.n_derivpoints):
#             K_mix = self._build_cov_dx2(
#                 self.X_data[:self.n_datapoints],
#                 self.X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim],
#                 index=i
#             )
#             K_NN = jnp.concatenate((K_NN,K_mix),axis=1)
#             sum_dims += dim
        
#         sum_dims_1 = 0
#         sum_dims_2 = 0
#         for i,dim_1 in enumerate(self.n_derivpoints):
#             K_mix = self._build_cov_dx2(
#                     self.X_data[:self.n_datapoints],
#                     self.X_data[self.n_datapoints+sum_dims_1:self.n_datapoints+sum_dims_1+dim_1],
#                     index=i
#                 )
#             for j,dim_2 in enumerate(self.n_derivpoints):
#                 K_derivs = self._build_cov_ddx1x2(
#                         self.X_data[self.n_datapoints+sum_dims_1:self.n_datapoints+sum_dims_1+dim_1],
#                         self.X_data[self.n_datapoints+sum_dims_2:self.n_datapoints+sum_dims_2+dim_2],
#                         index_1=i, index_2=j
#                     )
#                 K_mix = jnp.concatenate((K_mix,K_derivs),axis=0)
#                 sum_dims_2 += dim_2
#             K_NN = jnp.concatenate((K_NN,K_mix.T),axis=0)
#             sum_dims_1 += dim_1
#             sum_dims_2 = 0

#         return jnp.eye(len(self.X_data)) * (self.noise_var + self.diag_add) + K_NN
    
#     def fit(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray) -> None:
#         '''
#             Fits the full GPR Model

#             X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
#             Y_data.shape = (n_datapoints + sum(n_derivpoints),)
#         '''
#         self.X_data = X_data
#         self.Y_data = Y_data

#         self._fit_matrix = self.forward(X_data, Y_data)

#     # @partial(jit, static_argnums=(0,))
#     def predict(self, X: jnp.DeviceArray, return_std: bool = False):
#         if self._fit_matrix is None or self.Y_data is None:
#             raise ValueError("GPR not correctly fitted!")
        
#         full_vectors = self._build_cov_func(X, self.X_data[:self.n_datapoints])
#         sum_dims = 0
#         for i,dim in enumerate(self.n_derivpoints):
#             deriv_vectors = self._build_cov_dx2(
#                     X, 
#                     self.X_data[self.n_datapoints + sum_dims:self.n_datapoints + sum_dims + dim],
#                     index=i
#                 )
#             full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)
#             sum_dims += dim

#         means = full_vectors@solve(self._fit_matrix,self.Y_data)

#         if return_std:
#             X_cov = self._build_cov_vector(X)  
#             temp = self._build_xTAx(self._fit_matrix, full_vectors)      
#             stds = jnp.sqrt(X_cov - temp) # no noise term in the variance

#             return means, stds

#         return means

# class ApproximateGPR(BaseGPR):
#     def __init__(self, kernel =RBF(), n_datapoints: int = 0, n_derivpoints: Iterable = (0, ), X_ref: jnp.DeviceArray=None, noise_var: float = 0.000001) -> None:
#         '''
#             Approximates the full GP regressor by the Projected Process Approximation.
#         '''
#         super().__init__(kernel, n_datapoints, n_derivpoints, noise_var)

#         if X_ref is None:
#             raise ValueError("X_ref can't be None!")
#         self.X_ref = X_ref
#         self._K_ref = self._build_cov_func(X_ref, X_ref)

#     # @partial(jit, static_argnums=(0,))
#     def forward(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray):
#         K_MN = self._build_cov_func(self.X_ref, X_data[:self.n_datapoints])
#         sum_dims = 0
#         for i,dim in enumerate(self.n_derivpoints):
#             K_deriv = self._build_cov_dx2(
#                     self.X_ref, 
#                     X_data[self.n_datapoints+sum_dims:self.n_datapoints+sum_dims+dim],
#                     index=i
#             )
#             K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)
#             sum_dims += dim
            
#         # added small positive diagonal to make the matrix positive definite
#         fit_matrix = self.noise_var * self._K_ref + K_MN@K_MN.T + jnp.eye(len(self.X_ref)) * self.diag_add
#         fit_vector = K_MN@self.Y_data
        
#         return fit_matrix, fit_vector

#     def fit(self, X_data: jnp.DeviceArray, Y_data: jnp.DeviceArray):
#         '''
#             Fits the GPR based on the Projected Process Approximation

#             X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
#             Y_data.shape = (n_datapoints + sum(n_derivpoints),)
#         '''
#         self.X_data = X_data
#         self.Y_data = Y_data

#         self._fit_matrix, self._fit_vector = self.forward(X_data, Y_data)

#     # @partial(jit, static_argnums=(0,))
#     def predict(self, X: jnp.DeviceArray, return_std: bool = False):
        # '''
        #     Calculates the posterior means and standard deviations for the Projected Process Approximation

        #     X.shape = (N, n_features)
        # '''
        # if self._fit_matrix is None or self._fit_vector is None:
        #     raise ValueError("GPR not correctly fitted!")
        # if self.X_ref is None:
        #     raise ValueError("X_ref can't be None!")
        
        # ref_vectors = self._build_cov_func(self.X_ref, X).T

        # means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        # if return_std:
        #     X_cov = self._build_cov_vector(X)

        #     first_temp = self._build_xTAx(self._K_ref + jnp.eye(len(self.X_ref)) * self.diag_add, ref_vectors)
        #     second_temp = self.noise_var * self._build_xTAx(self._fit_matrix, ref_vectors)
            
        #     stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
            
        #     return means, stds
        
        # return means
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