import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit, vmap, Array
from functools import partial
from jaxopt import ScipyBoundedMinimize
from typing import Tuple, List

from .kernels import RBF

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
        self.fit_vector = None
        self.fit_matrix = None
        self.X_split = None

        # mode defines what happens when the class is called
        self.mode = None
        self.calls = {"regr": self._regression, "bays": self._bayesopt, "eval": self._eval}

    def __call__(self, *args, **kwargs):
        try:
            return self.calls[self.mode](*args, **kwargs)
        except KeyError:
            print("Current active mode not supported! \
                  Call training_regression(), training_bayesopt(), \
                  or eval() to set a supported mode.")

    # setters for the different modes in which to call the class
    def train_regression(self) -> None: self.mode = "regr"
    def train_bayesopt(self) -> None: self.mode = "bays"
    def eval(self) -> None: self.mode = "eval"

    # "virtual" methods that are to be defined in derived class
    def _regression(self, *args, **kwargs): raise NotImplementedError("Function hasn't been implemented in derived class")
    def _bayesopt(self, *args, **kwargs): raise NotImplementedError("Function hasn't been implemented in derived class")
    def _eval(self, *args, **kwargs): raise NotImplementedError("Function hasn't been implemented in derived class")
    
    def _negativeLogLikelyhood(self, params, *args):
        raise NotImplementedError("Forward method not yet implemented!")
    
    def optimize(self, init_params, *args):
        # solver = ProjectedGradient(self._min_obj, projection=projection_box)
        solver = ScipyBoundedMinimize(fun=self._negativeLogLikelyhood, method="l-bfgs-b")
        result = solver.run(init_params, (1e-3,jnp.inf), *args)

        # print(result)

        return result.params
    
    # @partial(vmap, in_axes=(None, None, 0))
    # def _build_xTAx(self, A, X):
    #     '''
    #         X.shape = (N,M)
    #         A.shape = (M,M)
    #         output.shape = (N,)
    #         Calculates x.T@A^-1@x for each of the N x in X (x.shape = (M,)).
    #     '''
    #     return X.T@solve(A,X,assume_a="pos")
    
    # def _build_cov_vector(self, X, params):
    #     '''
    #         X1.shape = (N, n_features)
    #         output.shape = (N,)
    #         Builds a vector of the covariance of all X[:,i] with them selves.
    #     '''
    #     func = lambda A: self.kernel.eval_func(A, A, params)
    #     func = vmap(func, in_axes=(0))
    #     return func(X)

    def _CovVector_Kernel(self, x, X, params):
        func = lambda A, B: self.kernel.eval_func(A, B, params)
        func = vmap(func, in_axes=(None,0))
        return func(x, X)

    def _CovVector_Grad(self, x, X, index, params):
        func = lambda A, B: self.kernel.eval_dx2(A, B, index, params)
        func = vmap(func, in_axes=(None,0))
        return func(x, X)
    
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
        self.X_split = None
        self.small_coeff = 1.0 / 20000.0

    def _regression(self, X_data: Array, Y_data: Array) -> None:
        '''
            Fit a sparse Gaussian Process to the given data

            X_data.shape = (n_samples, n_features)
            Y_data.shape = (n_samples, )
        '''
        # TODO: assertions and exceptions
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        self.X_split = jnp.split(X_data, sum_splits)

        self.params = self.optimize(self.params, Y_data, self.small_coeff, self.X_split)
        self.fit_matrix = self.forward(self.params, self.X_split)
        self.fit_vector = Y_data

    def _bayesopt(self):
        raise NotImplementedError

    def _eval(self, X: Array) -> Tuple[Array, Array]:
        '''
            Evaluate the posterior mean and standard deviation at each point in X

            X.shape = (n_samples, n_features)
        '''
        # TODO: assertions and exceptions
        # return self._meanstd_multi(X, self.params, self.fit_matrix, self.fit_vector, self.X_split)
        f = vmap(self._meanstd, in_axes=(0, None, None, None, None))
        return f(X, self.params, self.fit_matrix, self.fit_vector, self.X_split)

    @partial(jit, static_argnums=(0))
    def forward(self, params, X_split) -> Array:
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
        fit_matrix = jnp.eye(len(K_NN)) * (params[0]**2) + K_NN
        return fit_matrix
    
    @partial(jit, static_argnums=(0,))
    def _meanstd(self, x, params, fit_matrix, fit_vector, X_split):
        '''
            calculates the posterior mean & std for a single point x

            x.shape = (n_features,)
            X_split.shape = List[(N_i, n_features)]
                ... training data split into function evaluations and the different derivative evaluations
        '''
        # calculates covariance between x and all points in X_data 
        # depending on if the represent derivative or function evaluation
        cov_vector = self._CovVector_Kernel(x, X_split[0], params=params[1:])
        for i, elem in enumerate(X_split[1:]):
            temp = self._CovVector_Grad(x, elem, index=i, params=params[1:])
            cov_vector = jnp.hstack((cov_vector, temp))

        # calculates the covariance of x with itself
        cov_x = self.kernel.eval_func(x,x,params[1:])

        # calculates the full gp mean and standard deviation
        mean = cov_vector@solve(fit_matrix, fit_vector)
        var = cov_x - cov_vector@solve(fit_matrix, cov_vector)
        return mean, jnp.sqrt(var)
    
    # @partial(jit, static_argnums=(0,))
    # def _meanstd_multi(self, X, params, fitmatrix, fitvector, X_split):
    #     '''
    #         calculates the posterior mean & std for a set of points X

    #         x.shape = (n_features,)
    #         X_split.shape = List[(N_i, n_features)]
    #             ... training data split into function evaluations and the different derivative evaluations
    #     '''
    #     # calculates covariance between x and all points in X_data 
    #     # depending on if the represent derivative or function evaluation
    #     full_vectors = self._CovMatrix_Kernel(X, X_split[0], params=params[1:])
    #     for i,elem in enumerate(X_split[1:]):
    #         deriv_vectors = self._CovMatrix_KernelGrad(X, elem, index=i, params=params[1:])
    #         full_vectors = jnp.concatenate((full_vectors,deriv_vectors),axis=1)
            
    #     # calculates the covariance of X with itself
    #     X_cov = self._build_cov_vector(X, params[1:])  


    #     # calculates the full gp mean and standard deviation
    #     # mean = jnp.inner(cov_vector, solve(fit_matrix, fit_vector))
    #     # var = cov_x - jnp.inner(cov_vector, solve(fit_matrix, cov_vector))
    #     means = full_vectors@solve(fitmatrix,fitvector,assume_a="pos") 
    #     temp = self._build_xTAx(fitmatrix, full_vectors)     
    #     stds = jnp.sqrt(X_cov - temp) # no noise term in the variance
    #     return means, stds
    
    @partial(jit, static_argnums=(0,))
    def _negativeLogLikelyhood(self, params, Y_data, small_coeff, X_split) -> float:
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
        fit_matrix = self.forward(params, X_split)

        # calculates the logdet of the full covariance matrix
        _, logdet = jnp.linalg.slogdet(fit_matrix)

        # calculates Y.T@C**(-1)@Y and adds logdet to get the final result
        mle = logdet + Y_data@solve(fit_matrix, Y_data)
        
        return mle * small_coeff
    
    # @partial(jit, static_argnums=(0,))
    # def _mean(self, x, params, fit_matrix, fit_vector, X_split):
    #     '''
    #         calculates the posterior mean for a single point x

    #         x.shape = (n_features,)
    #         X_split.shape = List[(N_i, n_features)]
    #             ... training data split into function evaluations and the different derivative evaluations
    #     '''
    #     # calculate covariance between x and all points in X_data 
    #     # depending on if the represent derivative or function evaluation
    #     cov_vector = self._CovVector_Kernel(x, X_split[0], params=params[1:])
    #     for i, elem in enumerate(X_split[1:]):
    #         temp = self._CovVector_Grad(x, elem, index=i, params=params[1:])
    #         cov_vector = jnp.hstack((cov_vector, temp))

    #     # calculates the full gp mean
    #     mean = cov_vector@solve(fit_matrix, fit_vector)
    #     return mean
    
    # @partial(jit, static_argnums=(0,))
    # def _std(self, x, params, fit_matrix, X_split):
    #     '''
    #         calculates the posterior std for a single point x

    #         x.shape = (n_features,)
    #         X_split.shape = List[(N_i, n_features)]
    #             ... training data split into function evaluations and the different derivative evaluations
    #     '''
    #     # calculates covariance between x and all points in X_data 
    #     # depending on if the represent derivative or function evaluation
    #     cov_vector = self._CovVector_Kernel(x, X_split[0], params=params[1:])
    #     for i, elem in enumerate(X_split[1:]):
    #         temp = self._CovVector_Grad(x, elem, index=i, params=params[1:])
    #         cov_vector = jnp.hstack((cov_vector, temp))

    #     # calculates the covariance of x with itself
    #     cov_x = self.kernel.eval_func(x,x,params[1:])

    #     # calculates the full gp standard deviation
    #     # var = cov_x - jnp.inner(cov_vector, solve(fit_matrix, cov_vector))
    #     var = cov_x - cov_vector@solve(fit_matrix, cov_vector)
    #     return jnp.sqrt(var)
    
class SparseGPR(BaseGPR):
    def __init__(self, kernel=RBF(), data_split=(0,), X_ref=None, kernel_params=(1.0,), noise= 1e-4, *, noise_prior=None, kernel_prior=None) -> None:
        '''
            Sparsifies the full GP regressor by the Projected Process Approximation.
        '''
        super().__init__(kernel, data_split, kernel_params, noise, noise_prior=noise_prior, kernel_prior=kernel_prior)
        self.X_ref = X_ref
        self.small_coeff = 1.0 / 20000.0

    def _regression(self, X_data: Array, Y_data: Array) -> None:
        '''
            Fit a sparse Gaussian Process to the given data

            X_data.shape = (n_samples, n_features)
            Y_data.shape = (n_samples, )
        '''
        # TODO: assertions and exceptions
        sum_splits = [jnp.sum(self.data_split[:i+1]) for i,_ in enumerate(self.data_split[1:])]
        X_split = jnp.split(X_data, sum_splits)

        self.params = self.optimize(self.params, Y_data, self.small_coeff, self.X_ref, X_split)
        self.fit_matrix, self.fit_vector = self.forward(self.params, Y_data, self.X_ref, X_split)

    def _bayesopt(self):
        raise NotImplementedError

    def _eval(self, X: Array) -> Tuple[Array, Array]:
        '''
            Evaluate the posterior mean and standard deviation at each point in X

            X.shape = (n_samples, n_features)
        '''
        # TODO: assertions and exceptions
        # return self._meanstd_multi(X, self.params, self.fit_matrix, self.fit_vector, self.X_ref)
        f = vmap(self._meanstd, in_axes=(0, None, None, None, None))
        return f(X, self.params, self.fit_matrix, self.fit_vector, self.X_ref)

    @partial(jit, static_argnums=(0,))
    def forward(self, params, Y_data, X_ref, X_split) -> Tuple[Array, Array]:
        # calculates covariance matrix between reference points and data points
        K_MN = self._CovMatrix_Kernel(X_ref, X_split[0], params[1:])
        for i,elem in enumerate(X_split[1:]):
            K_deriv = self._CovMatrix_KernelGrad(X_ref, elem, index=i, params=params[1:])
            K_MN = jnp.concatenate((K_MN,K_deriv),axis=1)

        # calculates covariance matrix between reference points
        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])
            
        # added small positive diagonal to make the matrix positive definite
        fit_matrix = params[0]**2 * K_ref + K_MN@K_MN.T + jnp.eye(len(X_ref)) * 1e-6
        fit_vector = K_MN@Y_data

        return fit_matrix, fit_vector
    
    @partial(jit, static_argnums=(0,))
    def _meanstd(self, x, params, fitmatrix, fitvector, X_ref) -> Tuple[Array, Array]:
        '''
            calculates the posterior mean and std for a single point x
        '''
        # calculates covariance between new point and reference points
        ref_vector = self._CovVector_Kernel(x, X_ref, params[1:])

        # calculates covariance of the new point with itself
        cov_x = self.kernel.eval_func(x,x, params[1:])

        # calculates covariance between the reference points
        K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])
        # add a small positive value to the diagonal for numerical stability
        K_ref += jnp.eye(len(K_ref))*1e-6

        # calculate mean and variance according to PPA
        mean = ref_vector@solve(fitmatrix,fitvector)
        var = cov_x - ref_vector@solve(K_ref, ref_vector) \
                + params[0]**2 * (ref_vector@solve(fitmatrix, ref_vector))
        
        return mean, jnp.sqrt(var)
    
    # @partial(jit, static_argnums=(0,))
    # def _meanstd_multi(self, X, params, fitmatrix, fitvector, X_ref):
    #     '''
    #         calculates the posterior mean and std for a set of points X
    #     '''
    #     # calculates covariance between new points and reference points
    #     ref_vectors = self._CovMatrix_Kernel(X, X_ref, params[1:])

    #     # calculates covariance of the new points with themselves
    #     X_cov = self._build_cov_vector(X, params[1:])

    #     # calculates covariance between the reference points
    #     K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])
    #     # add a small positive value to the diagonal for numerical stability
    #     first_temp = self._build_xTAx(K_ref + jnp.eye(len(X_ref)) * 1e-6, ref_vectors)
    #     second_temp = params[0]**2 * self._build_xTAx(fitmatrix, ref_vectors)
        
    #     # calculate mean and variance according to PPA
    #     means = ref_vectors@solve(fitmatrix,fitvector)
    #     stds = jnp.sqrt(X_cov - first_temp + second_temp)
        
    #     return means, stds
    
    @partial(jit, static_argnums=(0,))
    def _negativeLogLikelyhood(self, params, Y_data, small_coeff, X_ref, X_split) -> float:
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
            K_deriv = self._CovMatrix_KernelGrad(X_ref, elem, index=i, params=params[1:])
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
        
        return mle * small_coeff
    
    # @partial(jit, static_argnums=(0,))
    # def _mean(self, x, params, fitmatrix, fitvector, X_ref) -> Array:
    #     '''
    #         calculates the posterior mean for a single point x

    #         x.shape = (n_features,)
    #         X_ref.shape = (n_refs, n_features)
    #     '''
    #     # vectorizes function over the n_refs dimension of X_ref
    #     f = vmap(lambda x, X: self.kernel.eval_func(x,X, params[1:]), in_axes=(None, 0))
    #     # calculates covariance between new point and reference points
    #     ref_vector = f(x,X_ref)
        
    #     # calculate mean according to PPA
    #     # mean = jnp.inner(ref_vector, solve(fitmatrix,fitvector))
    #     mean = ref_vector@solve(fitmatrix,fitvector)

    #     return mean
    
    # @partial(jit, static_argnums=(0,))
    # def _std(self, x, params, fitmatrix, X_ref) -> Array:
    #     '''
    #         calculates the posterior std for a single point x

    #         x.shape = (n_features,)
    #         X_ref.shape = (n_refs, n_features)
    #     '''
    #     # vectorizes function over the n_refs dimension of X_ref
    #     f = vmap(lambda x, X: self.kernel.eval_func(x,X, params[1:]), in_axes=(None, 0))
    #     # calculates covariance between new point and reference points
    #     ref_vector = f(x,X_ref)

    #     # calculates covariance of the new point with itself
    #     cov_x = self.kernel.eval_func(x,x, params[1:])

    #     # calculates covariance between the reference points
    #     K_ref = self._CovMatrix_Kernel(X_ref, X_ref, params=params[1:])

    #     # calculate variance according to PPA
    #     # var = cov_x - jnp.inner(ref_vector, solve(K_ref, ref_vector)) \
    #     #         + params[0]**2 * jnp.inner(ref_vector, solve(fitmatrix, ref_vector))
    #     var = cov_x - ref_vector@solve(K_ref, ref_vector) \
    #             + params[0]**2 * (ref_vector@solve(fitmatrix, ref_vector))

    #     return jnp.sqrt(var)