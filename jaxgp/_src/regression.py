import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit, vmap
from functools import partial
from jaxopt import ScipyBoundedMinimize

from .kernels import RBF
from .map import MaximumAPosteriori


class BaseGPR:
    def __init__(self, kernel=RBF(), data_split=(0, 0), init_kernel_params=(1.0, 1.0), noise=1e-4, *, noise_prior=None, kernel_prior=None) -> None:
        '''
            supported kernels are found in gaussion_process.kernels

            data_split : Iterable       ... first element:      number of (noisy) function evaluations
                                            further elements:   number of (noisy) derivative evaluations
            kernel_params : Iterable    ... initial parameters for the MLE parameter optimization
            noise_var : float           ... assumed noise in the evaluation of data
        '''
        self.kernel = kernel
        self.params = jnp.array((noise,) + init_kernel_params)
        self.data_split = jnp.array(data_split)

        # initialized variables to save from the fitting step
        self._fit_vector = None
        self._fit_matrix = None
        self.forward_args = []

        # optimizer
        self.mle = MaximumAPosteriori()

    def train(self, X_data, Y_data) -> None:
        '''
            Fits the GPR Model to the given data

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)

            Thin wrapper with all side effects around the pure functions 
            self.optimize and self.forward that do the actual work.
        '''
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        self.forward_args.append(jnp.split(X_data, sum_splits))

        self.params = self.optimize(self.params, Y_data, *self.forward_args)
        self._fit_matrix, self._fit_vector = self.forward(self.params, Y_data, *self.forward_args)

    def forward(self, params, Y_data, *args):
        raise NotImplementedError("Forward method not yet implemented!")
    
    def eval(self, X, return_std=False):
        '''
            Predicts the posterior mean (and std if return_std=True) at each point in X
            X.shape = (N, n_features)
            
            Thin wrapper with all side effects around the pure function 
            self.eval that does the actual work.
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        
        if return_std:
            return self.mean_std_eval(self.params, X, self._fit_matrix, self._fit_vector, *self.forward_args)
        
        return self.mean_eval(self.params, X, self._fit_matrix, self._fit_vector, *self.forward_args)
    
    def mean_eval(self, params, X, fitmatrix, fitvector, *args):
        raise NotImplementedError("Forward method not yet implemented!")
        
    def mean_std_eval(self, params, X, fitmatrix, fitvector, *args):
        raise NotImplementedError("Forward method not yet implemented!")
    
    def _min_obj(self, params, *args):
        raise NotImplementedError("Forward method not yet implemented!")
 
    # @partial(jit, static_argnums=(0))
    def testfunction(self, init_params, *args):
        # num_steps = 80

        # minimum = jnp.array([jnp.inf, jnp.inf, jnp.inf])

        # grid = jnp.linspace(1e-4, 5.0, num_steps)

        # for noise in grid:
        #     for ls in grid:
        #         next_try = self._min_obj(jnp.array([noise, ls]), *args)
        #         minimum = jnp.where(next_try < minimum[0], jnp.array([next_try, noise, ls]), minimum)
        
        num_steps = 50
        grid = jnp.linspace(1e-4, 3.0, num_steps)

        best_params = jnp.array(init_params)
        old_params = jnp.ones_like(best_params)*jnp.inf

        while(jnp.sum(jnp.abs(old_params-best_params)) > 1e-3):
            old_params = best_params

            new_params = []

            for i,_ in enumerate(best_params):
                minimum = jnp.array([jnp.inf, jnp.inf])

                for param in grid:
                    next_try = self._min_obj(jnp.array([*best_params[:i], param, *best_params[i+1:]]), *args)
                    minimum = jnp.where(next_try < minimum[0], jnp.array([next_try, param]), minimum)

                new_params.append(minimum[1])

            best_params = (jnp.array(new_params) + best_params) / 2
            # print(f"{old_params=}", f"{best_params=}")
        

        return best_params
        # def loop_ls(i, val):
        #     temp = self._min_obj(jnp.array([*left_params, 0.05*i, *right_params]), *args)
        #     return jnp.where(val[1] < temp, val, jnp.array([0.05*i,temp]))
        
        # val = jnp.array([0.0, self._min_obj(jnp.array([*left_params, 1e-4, *right_params]), *args)])
        
        # result = fori_loop(1, 100, loop_ls, val)

        # return result
    
    def optimize(self, init_params, *args):
        # solver = ProjectedGradient(self._min_obj, projection=projection_box)
        solver = ScipyBoundedMinimize(fun=self._min_obj, method="l-bfgs-b")
        result = solver.run(init_params, (1e-3,jnp.inf), *args)

        print(result)

        # test = self.testfunction(init_params, *args)
        # print(test)

        return result.params
        
        raise ValueError("Optimization failed!")
    
    # @partial(jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, None, 0))
    def _build_xTAx(self, A, X):
        '''
            X.shape = (N,M)
            A.shape = (M,M)

            output.shape = (N,)

            Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
        '''
        return X.T@solve(A,X,assume_a="pos")
    
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
    def __init__(self, kernel=RBF(), data_split=(0, 0), kernel_params=(1.0,), noise=1e-4, *, noise_prior=None, kernel_prior=None) -> None:
        super().__init__(kernel, data_split, kernel_params, noise, noise_prior=noise_prior, kernel_prior=kernel_prior)

        self.forward_args = []

    @partial(jit, static_argnums=(0))
    def forward(self, params, Y_data, X_split):
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
        return (jnp.eye(len(K_NN)) * (params[0]**2 + 1e-6) + K_NN), (Y_data)
    
    @partial(jit, static_argnums=(0,))
    def mean_eval(self, params, X, fitmatrix, fitvector, X_split):
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_KernelGrad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector,assume_a="pos")
        
        return means
    
    @partial(jit, static_argnums=(0,))
    def mean_std_eval(self, params, X, fitmatrix, fitvector, X_split):
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_KernelGrad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector,assume_a="pos")

        X_cov = self._build_cov_vector(X, params[1:])  
        temp = self._build_xTAx(fitmatrix, full_vectors)      
        stds = jnp.sqrt(X_cov - temp) # no noise term in the variance
        
        return means, stds

    @partial(jit, static_argnums=(0))
    def _min_obj(self, params, *args):
        fitmatrix, fitvector = self.forward(params, *args)
        return -self.mle.forward(params, fitmatrix, fitvector) / 5000.0

class SparseGPR(BaseGPR):
    def __init__(self, kernel=RBF(), data_split=(0,), X_ref=None, kernel_params=(1.0,), noise= 1e-4, *, noise_prior=None, kernel_prior=None) -> None:
        '''
            Sparsifies the full GP regressor by the Projected Process Approximation.
        '''
        super().__init__(kernel, data_split, kernel_params, noise, noise_prior=noise_prior, kernel_prior=kernel_prior)

        self.forward_args = [X_ref, ]

    @partial(jit, static_argnums=(0,))
    def forward(self, params, Y_data, X_ref, X_split):
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_KernelGrad(X_ref, elem, index=i, params=params[1:])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = params[0]**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-6
        fit_vector = K_MN@Y_data

        return fit_matrix, fit_vector
    
    @partial(jit, static_argnums=(0,))
    def mean_eval(self, params, X, fitmatrix, fitvector, X_ref, X_split):
        ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])

        means = ref_vectors@solve(fitmatrix,fitvector,assume_a="pos")
        
        return means
    
    @partial(jit, static_argnums=(0,))
    def mean_std_eval(self, params, X, fitmatrix, fitvector, X_ref, X_split):
        ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])

        means = ref_vectors@solve(fitmatrix,fitvector,assume_a="pos")

        X_cov = self._build_cov_vector(X, params[1:])

        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

        first_temp = self._build_xTAx(K_ref + jnp.eye(len(X_ref)) * 1e-6, ref_vectors)
        second_temp = params[0]**2 * self._build_xTAx(fitmatrix, ref_vectors)
        
        stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
        
        return means, stds

    @partial(jit, static_argnums=(0))
    def _min_obj(self, params, Y_data, X_ref, X_split):
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_KernelGrad(X_ref, elem, index=i, params=params[1:])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

        fitmatrix = jnp.eye(len(Y_data)) * (1e-6 + params[0]**2) + K_MN.T@solve(K_ref,K_MN, assume_a="pos")

        return -self.mle.forward(params, fitmatrix, Y_data) / 5000.0