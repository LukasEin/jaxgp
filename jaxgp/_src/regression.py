import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit, vmap
from functools import partial
from jaxopt import ScipyBoundedMinimize

from jax import Array
from typing import Union, Tuple

from .kernels import RBF

class BaseGPR:
    def __init__(self, kernel=RBF(), data_split=(0, 0), init_kernel_params=(1.0, 1.0), noise=1e-4, *, optimize_noise=False) -> None:
        '''
            supported kernels are found in gaussion_process.kernels

            data_split : Iterable       ... first element:      number of (noisy) function evaluations
                                            further elements:   number of (noisy) derivative evaluations
            kernel_params : Iterable    ... initial parameters for the MLE parameter optimization
            noise_var : float           ... assumed noise in the evaluation of data
        '''
        self.kernel = kernel
        self.data_split = jnp.array(data_split)

        # defines parameters of the regressor and sets flag for optimizing the noise parameter as well
        self.optimize_noise = optimize_noise
        self.kernel_params = init_kernel_params
        self.noise = noise
        self.params = jnp.array((noise,) + init_kernel_params)

        # initialized variables to save from the fitting step
        self.fit_vector = None
        self.fit_matrix = None
        self.further_args = []

    def train(self, X_data: Array, Y_data: Array) -> None:
        '''
            Fits the GPR Model to the given data

            X_data.shape = (n_datapoints + sum(n_derivpoints), n_features)
            Y_data.shape = (n_datapoints + sum(n_derivpoints),)

            Thin wrapper with all side effects around the pure functions 
            self.optimize and self.forward that do the actual work.
        '''
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        self.further_args.append(jnp.split(X_data, sum_splits))

        self.optimize(Y_data, *self.further_args)
        self.fit_matrix, self.fit_vector = self.forward(self.params, Y_data, *self.further_args)

    def forward(self, params: Array, Y_data: Array, *further_args) -> Tuple[Array, Array]:
        raise NotImplementedError("Forward method not yet implemented!")
    
    def eval(self, X: Array, return_std=False) -> Union[Tuple[Array, Array], Array]:
        '''
            Predicts the posterior mean (and std if return_std=True) at each point in X
            X.shape = (N, n_features)
            
            Thin wrapper with all side effects around the pure function 
            self.eval that does the actual work.
        '''
        if self.fit_matrix is None or self.fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        
        if return_std:
            return self._meanstd(self.params, X, self.fit_matrix, self.fit_vector, *self.further_args)
        
        return self._mean(self.params, X, self.fit_matrix, self.fit_vector, *self.further_args)
    
    def _mean(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, *further_args) -> Array: raise NotImplementedError
    def _meanstd(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, *further_args) -> Tuple[Array, Array]: raise NotImplementedError
    def _negativeLogLikelyhood(self, params: Array, Y_data: Array, *further_args) -> float: raise NotImplementedError
    def _kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, *further_args) -> float: raise NotImplementedError
    
    def optimize(self, Y_data: Array, *further_args) -> None:
        '''
            Maximizes the log likelyhood function either for 
                the kernel parameters or
                the noise and the kernel parameters
        '''
        if self.optimize_noise:
            solver = ScipyBoundedMinimize(fun=self._negativeLogLikelyhood, method="l-bfgs-b")
            result = solver.run(self.params, (1e-3,jnp.inf), Y_data, *further_args)

            self.params = result.params
        else:
            solver = ScipyBoundedMinimize(fun=self._kernelNegativeLogLikelyhood, method="l-bfgs-b")
            result = solver.run(self.kernel_params, (1e-3,jnp.inf), self.noise, Y_data, *further_args)


            self.params = jnp.array((self.noise, ) + result.params)
    
    @partial(vmap, in_axes=(None, None, 0))
    def _build_xTAx(self, A: Array, X: Array) -> Array:
        '''
            X.shape = (N,M)
            A.shape = (M,M)

            output.shape = (N,)

            Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
        '''
        return X.T@solve(A,X,assume_a="pos")
    
    def _CovVector_Id(self, X: Array, params: Array) -> Array:
        '''
            X1.shape = (N, n_features)

            output.shape = (N,)

            Builds a vector of the covariance of all X[:,i] with them selves.
        '''
        func = lambda A: self.kernel.eval_func(A, A, params)
        func = vmap(func, in_axes=(0))
        return func(X)

    def _CovMatrix_Kernel(self, X1: Array, X2: Array, params: Array) -> Array:
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
    
    def _CovMatrix_Grad(self, X1: Array, X2: Array, index: int, params: Array) -> Array:
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
    
    def _CovMatrix_Hess(self, X1: Array, X2: Array, index_1: int, index_2: int, params: Array) -> Array:
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
    def __init__(self, kernel=RBF(), data_split=(0, 0), kernel_params=(1.0,), noise=1e-4, *, optimize_noise=False, noise_prior=None, kernel_prior=None) -> None:
        super().__init__(kernel, data_split, kernel_params, noise, noise_prior=noise_prior, kernel_prior=kernel_prior, optimize_noise=optimize_noise)

        self.further_args = []

    @partial(jit, static_argnums=(0))
    def forward(self, params: Array, Y_data: Array, X_split: list[Array]) -> Tuple[Array, Array]:
        # Build the full covariance Matrix between all datapoints in X_data depending on if they   
        # represent function evaluations or derivative evaluations
        K_NN = self._CovMatrix_Kernel(X_split[0], X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_mix = self._CovMatrix_Grad(X_split[0], elem, index=i, params=params[1:])
            K_NN = jnp.concatenate((K_NN,K_mix),axis=1)
        
        for i,elem_1 in enumerate(X_split[1:]):
            K_mix = self._CovMatrix_Grad(X_split[0], elem_1, index=i, params=params[1:])

            for j,elem_2 in enumerate(X_split[1:]):
                K_derivs = self._CovMatrix_Hess(elem_1, elem_2, index_1=i, index_2=j, params=params[1:])
                K_mix = jnp.concatenate((K_mix,K_derivs),axis=0)

            K_NN = jnp.concatenate((K_NN,K_mix.T),axis=0)

        # additional small diagonal element added for 
        # numerical stability of the inversion and determinant
        return (jnp.eye(len(K_NN)) * (params[0]**2 + 1e-6) + K_NN), (Y_data)
    
    @partial(jit, static_argnums=(0,))
    def _mean(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, X_split: Array) -> Array:
        '''
            calculates the posterior mean for all points in X
        '''
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_Grad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector,assume_a="pos")
        
        return means
    
    @partial(jit, static_argnums=(0,))
    def _meanstd(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, X_split: Array) -> Tuple[Array, Array]:
        '''
            calculates the posterior mean and std for all points in X
        '''
        full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
        for i,elem in enumerate(X_split[1:]):
            deriv_vectors = self._CovMatrix_Grad(X, elem, index=i, params=params[1:])
            full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)

        means = full_vectors@solve(fitmatrix,fitvector,assume_a="pos")

        X_cov = self._CovVector_Id(X, params[1:])  
        temp = self._build_xTAx(fitmatrix, full_vectors)      
        stds = jnp.sqrt(X_cov - temp) # no noise term in the variance
        
        return means, stds

    @partial(jit, static_argnums=(0))
    def _negativeLogLikelyhood(self, params: Array, Y_data: Array, X_split: list[Array]) -> float:
        '''
            for PPA the Y_data ~ N(0,[id*s**2 + K_NN])
                which is the same as for Nystrom approximation

            The negative log likelyhood estimate is calculated according 
                to this distribution.

            Everything that is unnecessary to calculate the minimum has been removed

            Formally calculates:
            log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 - K_NN

            result is multiplied with a small coefficient for numerical stability 
                of the optimizer
        '''
        # calculates the full covariance matrix
        fit_matrix, fit_vector = self.forward(params, Y_data, X_split)
        fit_vector = fit_vector.reshape(-1)

        # calculates the logdet of the full covariance matrix
        _, logdet = jnp.linalg.slogdet(fit_matrix)

        # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
        mle = logdet + fit_vector@solve(fit_matrix, fit_vector, assume_a="pos")
        
        return mle / 20000.0
    
    @partial(jit, static_argnums=(0,))
    def _kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, X_split: list[Array]) -> float:
        '''
            for PPA the Y_data ~ N(0,[id*s**2 + K_NN])
                which is the same as for Nystrom approximation

            The negative log likelyhood estimate is calculated according 
                to this distribution.

            Everything that is unnecessary to calculate the minimum has been removed

            Formally calculates:
            log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 - K_NN

            result is multiplied with a small coefficient for numerical stability 
                of the optimizer
        '''
        params = jnp.array((noise,) + kernel_params)
        # calculates the full covariance matrix
        fit_matrix, fit_vector = self.forward(params, Y_data, X_split)
        fit_vector = fit_vector.reshape(-1)

        # calculates the logdet of the full covariance matrix
        _, logdet = jnp.linalg.slogdet(fit_matrix)

        # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
        mle = logdet + fit_vector@solve(fit_matrix, fit_vector, assume_a="pos")
        
        return mle / 20000.0

class SparseGPR(BaseGPR):
    def __init__(self, kernel=RBF(), data_split=(0,), X_ref=None, kernel_params=(1.0,), noise= 1e-4, *, optimize_noise=False, noise_prior=None, kernel_prior=None) -> None:
        '''
            Sparsifies the full GP regressor by the Projected Process Approximation.
        '''
        super().__init__(kernel, data_split, kernel_params, noise, optimize_noise=optimize_noise, noise_prior=noise_prior, kernel_prior=kernel_prior)

        self.further_args = [X_ref, ]

    @partial(jit, static_argnums=(0,))
    def forward(self, params: Array, Y_data: Array, X_ref: Array, X_split: list[Array]) -> Tuple[Array, Array]:
        # calculates the covariance between the training points and the reference points
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_Grad(X_ref, elem, index=i, params=params[1:])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        # calculates the covariance between each pair of reference points
        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = params[0]**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-4
        fit_vector = K_MN@Y_data
        return fit_matrix, fit_vector
    
    @partial(jit, static_argnums=(0,))
    def _mean(self, params: Array, X, fitmatrix: Array, fitvector: Array, X_ref: Array, X_split: list[Array]) -> Array:
        '''
            calculates the posterior mean for all points in X
        '''
        # calculates the caovariance between the test points and the reference points
        ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])
        
        means = ref_vectors@solve(fitmatrix,fitvector)#,assume_a="pos")
        return means
    
    @partial(jit, static_argnums=(0,))
    def _meanstd(self, params: Array, X: Array, fitmatrix: Array, fitvector: Array, X_ref: Array, X_split: list[Array]) -> Tuple[Array, Array]:
        '''
            calculates the posterior mean and std for all points in X
        '''
        # calculates the caovariance between the test points and the reference points
        ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])

        means = ref_vectors@solve(fitmatrix,fitvector)#,assume_a="pos")

        X_cov = self._CovVector_Id(X, params[1:])

        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

        first_temp = self._build_xTAx(K_ref + jnp.eye(len(X_ref)) * 1e-4, ref_vectors)
        second_temp = params[0]**2 * self._build_xTAx(fitmatrix, ref_vectors)
        
        stds = jnp.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
        
        return means, stds
    
    def _negativeLogLikelyhood(self, params: Array, Y_data: Array, X_ref: Array, X_split: list[Array]) -> float:
        '''
            for PPA the Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN])
                which is the same as for Nystrom approximation

            The negative log likelyhood estimate is calculated according 
                to this distribution.

            Everything that is unnecessary to calculate the minimum has been removed

            Formally calculates:
            log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 + K_MN.T@K_MM**(-1)@K_MN

            result is multiplied with a small coefficient for numerical stability 
                of the optimizer
        '''
        # calculates the covariance between the data and the reference points
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_Grad(X_ref, elem, index=i, params=params[1:])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        # calculates the covariance between the reference points
        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

        # directly calculates the logdet of the Nystrom covariance matrix
        log_matrix = jnp.eye(len(Y_data)) * (1e-6 + params[0]**2) + K_MN.T@solve(K_ref,K_MN)
        _, logdet = jnp.linalg.slogdet(log_matrix)

        # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
        invert_matrix = K_ref*params[0]**2 + K_MN@K_MN.T
        mle = logdet + (Y_data@Y_data -
                        Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / params[0]**2
        
        return mle / 20000.0
    
    @partial(jit, static_argnums=(0,))
    def _kernelNegativeLogLikelyhood(self, kernel_params: Array, noise: Union[Array, float], Y_data: Array, X_ref: Array, X_split: list[Array]) -> float:
        '''
            for PPA the Y_data ~ N(0,[id*s**2 + K_MN.T@K_MM**(-1)@K_MN])
                which is the same as for Nystrom approximation

            The negative log likelyhood estimate is calculated according 
                to this distribution.

            Everything that is unnecessary to calculate the minimum has been removed

            Formally calculates:
            log(det(C)) + Y.T@C**(-1)@Y, C := id*s**2 + K_MN.T@K_MM**(-1)@K_MN

            result is multiplied with a small coefficient for numerical stability 
                of the optimizer
        '''
        # calculates the covariance between the data and the reference points
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], kernel_params)
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_Grad(X_ref, elem, index=i, params=kernel_params)
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        # calculates the covariance between the reference points
        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=kernel_params)

        # directly calculates the logdet of the Nystrom covariance matrix
        log_matrix = jnp.eye(len(Y_data)) * (1e-6 + noise**2) + K_MN.T@solve(K_ref,K_MN)
        _, logdet = jnp.linalg.slogdet(log_matrix)

        # efficiently calculates Y.T@C**(-1)@Y and adds logdet to get the final result
        invert_matrix = K_ref*noise**2 + K_MN@K_MN.T
        mle = logdet + (Y_data@Y_data -
                        Y_data@K_MN.T@solve(invert_matrix, K_MN@Y_data)) / noise**2
        
        return mle / 20000.0