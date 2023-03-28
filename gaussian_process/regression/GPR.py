import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.optimize import minimize
from ..kernels import RBF
from jax import jit, vmap, grad
from functools import partial
import matplotlib.pyplot as plt


class BaseGPR:
    def __init__(self, kernel=RBF(), data_split=(0, 0), init_kernel_params=(1.0, 1.0), noise_var=1e-6) -> None:
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
        self.X_split = None

    # @partial(jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, None, 0))
    def _build_xTAx(self, A, X):
        '''
            X.shape = (N,M)
            A.shape = (M,M)

            output.shape = (N,)

            Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
        '''
        return X.T@solve(A,X)
    
    # @partial(jit, static_argnums=(0,))
    # @partial(vmap, in_axes=(None,0))
    def _build_cov_vector(self, X, params):
        '''
            X1.shape = (N, n_features)

            output.shape = (N,)

            Builds a vector of the covariance of all X[:,i] with them selves.
        '''
        func = lambda A: self.kernel.eval_func(A, A, params)
        func = vmap(func, in_axes=(0))
        return func(X)

    # @partial(jit, static_argnums=(0,))
    def _CovMatrix_Kernel(self, X1, X2, params):
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
    def _CovMatrix_KernelGrad(self, X1, X2, index, params):
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
    def _CovMatrix_KernelHess(self, X1, X2, index_1, index_2, params):
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
    def __init__(self, kernel=RBF(), data_split=(0, 0), kernel_params=(1.0, 1.0), noise_var=1e-6) -> None:
        super().__init__(kernel, data_split, kernel_params, noise_var)
    
    @partial(jit, static_argnums=(0))
    def LogLikelyhoodEstimate(self, params, X_split, Y_data):
        fitmatrix, fitvector = self.forward(X_split, Y_data, params)
        _, logdet = jnp.linalg.slogdet(fitmatrix)
        fitvector = fitvector.reshape(-1)
        return -0.5*(len(fitvector) * jnp.log(2*jnp.pi) + logdet + fitvector.T@solve(fitmatrix,fitvector))

    def optimize(self, initial_params, X_split, Y_data):
        # result = jit(minimize(self.negativeLogLikelyhoodEstimate, initial_params, (X_data, Y_data, data_split), method="BFGS"))
        # result = minimize(self.negativeLogLikelyhoodEstimate, initial_params, (X_split, Y_data), method="BFGS")

        # if result.success:
        #     if result.status == 1:
        #         print("Max iterations reached!")
        #         return result.x
        #     if result.status == 0:
        #         return result.x
            
        # raise ValueError("An error occured while maximizing the log likelyhood")
        
        grids = [jnp.linspace(0.5,5.0,50),jnp.linspace(0.5,10.0,50)]
        mesh = jnp.array(jnp.meshgrid(*grids)).reshape(2,2500).T
        params = jnp.array([[elem[0],1.0,elem[1]] for elem in mesh])
        mle = jnp.array([self.LogLikelyhoodEstimate(param,X_split,Y_data) for param in params]).reshape(50,50)

        plt.pcolormesh(*grids,mle)
        plt.colorbar()

        # grid = jnp.linspace(0.01,5.0)
        # params = jnp.array([[elem,1.0,1.0] for elem in grid])
        # mle_1 = jnp.array([self.LogLikelyhoodEstimate(param,X_split,Y_data) for param in params])
        # params = jnp.array([[0.1,elem,1.0] for elem in grid])
        # mle_2 = jnp.array([self.LogLikelyhoodEstimate(param,X_split,Y_data) for param in params])
        # params = jnp.array([[0.1,1.0,elem] for elem in grid])
        # mle_3 = jnp.array([self.LogLikelyhoodEstimate(param,X_split,Y_data) for param in params])
        
        # plt.plot(grid,mle_1,label="noise")
        # plt.plot(grid,mle_2,label="pre_coeff")
        # plt.plot(grid,mle_3,label="lengthscale")
        # plt.legend()

        return initial_params
        
    def fit(self, X_data, Y_data) -> None:
        '''
            Fits the GPR Model to the given data

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)

            Thin wrapper with all side effects around the pure functions 
            self.optimize and self.forward that do the actual work.
        '''        
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        self.X_split = jnp.split(X_data, sum_splits)

        self.params = self.optimize(self.params, self.X_split, Y_data)
        self._fit_matrix, self._fit_vector = self.forward(self.X_split, Y_data, self.params)

    @partial(jit, static_argnums=(0))
    def forward(self, X_split, Y_data, params):
        # Build the full covariance Matrix between all datapoints in X_data depending on if they   
        # represent function evaluations or derivative evaluations
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
        return (jnp.eye(len(K_NN)) * (params[0] + 1e-6) + K_NN), (Y_data)
    
    def predict(self, X, return_std=False):
        '''
            Predicts the posterior mean (and std if return_std=True) at each point in X
            X.shape = (N, n_features)
            
            Thin wrapper with all side effects around the pure function 
            self.eval that does the actual work.
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        
        if return_std:
            return self.eval_mean_std(X, self.X_split, self._fit_matrix, self._fit_vector, self.params)
        
        return self.eval_mean(X, self.X_split, self._fit_matrix, self._fit_vector, self.params, return_std)
    
    @partial(jit, static_argnums=(0,))
    def eval_mean(self, X, X_split, fitmatrix, fitvector, params):
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_KernelGrad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector)
        
        return means
    
    @partial(jit, static_argnums=(0,))
    def eval_mean_std(self, X, X_split, fitmatrix, fitvector, params):
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_KernelGrad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector)

        X_cov = self._build_cov_vector(X, params[1:])  
        temp = self._build_xTAx(fitmatrix, full_vectors)      
        stds = jnp.sqrt(X_cov - temp) # no noise term in the variance
        
        return means, stds

class ApproximateGPR(BaseGPR):
    def __init__(self, kernel=RBF(), data_split=(0, ), X_ref=None, kernel_params= (1.0, 1.0), noise_var= 1e-6) -> None:
        '''
            Approximates the full GP regressor by the Projected Process Approximation.
        '''
        super().__init__(kernel, data_split, kernel_params, noise_var)

        self.X_ref = X_ref
        self._K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=self.params[1:])
    
    def LogLikelyhoodEstimate(self, params, X_split, Y_data, X_ref, K_ref):
        fitmatrix, fitvector = self.forward(X_split, Y_data, X_ref, K_ref, params)
        _, logdet = jnp.linalg.slogdet(fitmatrix)
        fitvector = fitvector.reshape(-1)
        return -0.5*(len(fitvector) * jnp.log(2*jnp.pi) + logdet + fitvector.T@solve(fitmatrix,fitvector))

    def optimize(self, initial_params, X_split, Y_data, X_ref, K_ref):
        # result = jit(minimize(self.negativeLogLikelyhoodEstimate, initial_params, (X_data, Y_data, data_split), method="BFGS"))
        # result = minimize(self.LogLikelyhoodEstimate, initial_params, (X_split, Y_data, X_ref, K_ref), method="BFGS")

        # print(result)

        # if result.success:
        #     if result.status == 1:
        #         print("Max iterations reached!")
        #         return result.x
        #     if result.status == 0:
        #         return result.x
            
        # raise ValueError("An error occured while maximizing the log likelyhood")

        grid = jnp.linspace(0.01,1.0)
        params = jnp.array([[elem,1.0,1.0] for elem in grid])
        mle = jnp.array([self.LogLikelyhoodEstimate(param) for param in params])
        
        plt.plot(grid,mle)
        plt.savefig("test.png")
   
    def fit(self, X_data, Y_data) -> None:
        '''
            Fits the GPR Model to the given data

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)

            Thin wrapper with all side effects around the pure functions 
            self.optimize and self.forward that do the actual work.
        '''
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        self.X_split = jnp.split(X_data, sum_splits)

        # self.params = self.optimize(self.params, self.X_split, Y_data, self.X_ref, self._K_ref)
        self._fit_matrix, self._fit_vector = self.forward(self.X_split, Y_data, self.X_ref, self._K_ref, self.params)

    @partial(jit, static_argnums=(0,))
    def forward(self, X_split, Y_data, X_ref, K_ref, params):
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_KernelGrad(X_ref, elem, index=i, params=params[1:])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = params[0] * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-6
        fit_vector = K_MN@Y_data

        return fit_matrix, fit_vector

    def predict(self, X, return_std=False):
        '''
            Predicts the posterior mean (and std if return_std=True) at each point in X
            X.shape = (N, n_features)
            
            Thin wrapper with all side effects around the pure function 
            self.eval that does the actual work.
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        
        if return_std:
            return self.eval_mean_std(X, self.X_ref, self._K_ref, self._fit_matrix, self._fit_vector, self.params)
        
        return self.eval_mean(X, self.X_ref, self._fit_matrix, self._fit_vector, self.params, return_std)
    
    @partial(jit, static_argnums=(0,))
    def eval_mean(self, X, X_ref, fitmatrix, fitvector, params):
        ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])

        means = ref_vectors@solve(fitmatrix,fitvector)
        
        return means
    
    @partial(jit, static_argnums=(0,))
    def eval_mean_std(self, X, X_ref, K_ref, fitmatrix, fitvector, params):
        ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])

        means = ref_vectors@solve(fitmatrix,fitvector)

        X_cov = self._build_cov_vector(X, params[1:])

        first_temp = self._build_xTAx(K_ref + jnp.eye(len(X_ref)) * 1e-6, ref_vectors)
        second_temp = params[0] * self._build_xTAx(fitmatrix, ref_vectors)
        
        stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
        
        return means, stds