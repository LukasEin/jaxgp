import numpy as np
from scipy.linalg import solve

class sparseGPR_1D:
    def __init__(self, kernel, num_datapoints=None, num_derivpoints=None, X_ref=None, *, sparse_method="ppa", noise_var=1e-6) -> None:
        self.kernel = kernel

        self.num_datapoints = num_datapoints
        self.num_derivpoints = num_derivpoints

        # X_ref must be given for now!!!!!
        if X_ref is None:
            raise ValueError("X_ref can't be None!")
        self.X_ref = X_ref
        self.num_referencepoints = len(X_ref)
        self._K_ref = self._create_covMatrix(self.X_ref,self.X_ref,self.kernel.eval)

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
            X_data ... (datapoint_1, ..., datapoint_N1, derivpoint_1, ..., derivpoint_N2)
            Y_data ... same shape as X_data
            
            for now X_ref must be given!!!!
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
            for now X_ref must be given!!!
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

            X_data ...  (datapoint_1, ..., datapoint_N1, derivpoint_1, ..., derivpoint_N2)
                        N1 must equal self.num_datapoints and N2 must equal self.num_derivpoints
            Y_data ...  same shape as X_data
        '''
        X_func = self.X_data[:self.num_datapoints]
        X_deriv = self.X_data[self.num_datapoints:]
        
        K_func = self._create_covMatrix(self.X_ref,X_func,self.kernel.eval)
        K_deriv = self._create_covMatrix(self.X_ref,X_deriv,self.kernel.dx2)
        K_MN = np.concatenate((K_func,K_deriv),axis=0).T

        # added small positive diagonal to make the matrix positive definite
        self._fit_matrix = self.noise_var * self._K_ref + K_MN@K_MN.T + np.eye(len(self.X_ref))*1e-10
        self._fit_vector = K_MN@self.Y_data

    def _fit_nystrom(self) -> None:
        '''
            Fits the GPR based on the Nystroem Method

            X_data ...  (datapoint_1, ..., datapoint_N1, derivpoint_1, ..., derivpoint_N2)
                        N1 must equal self.num_datapoints and N2 must equal self.num_derivpoints
            Y_data ...  same shape as X_data
        '''
        K_func = self._create_covMatrix(self.X_ref,self.X_data[:self.num_datapoints],self.kernel.eval)
        K_deriv = self._create_covMatrix(self.X_ref,self.X_data[self.num_datapoints:],self.kernel.dx2)
        K_MN = np.concatenate((K_func,K_deriv),axis=0).T
        # K_MN = self._create_covMatrix(self.X_ref,self.X_data,self.kernel.eval).T

        # added small positive diagonal to make the matrix positive definite
        temp = self.noise_var * self._K_ref + K_MN@K_MN.T + np.eye(len(self.X_ref))*1e-10
        self._fit_matrix = (np.eye(len(self.X_data)) - K_MN.T@solve(temp,K_MN)) / self.noise_var

    def _predict_PPA(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Projected Process Approximation
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        ref_vectors = self._create_covMatrix(self.X_ref,X,self.kernel.eval)

        means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        if return_std:
            X_cov = self.kernel.eval(X,X)
            first_temp = np.array([k.T@solve(self._K_ref + np.eye(len(self.X_ref))*1e-10,k) for k in ref_vectors])
            second_temp =  self.noise_var * np.array([k.T@solve(self._fit_matrix,k) for k in ref_vectors])
            
            # stds = np.sqrt(X_cov + self.noise_var - first_temp + second_temp)
            stds = np.sqrt(X_cov - first_temp + second_temp) # no noise term in the variance
            
            return means, stds
        
        return means

    def _predict_SoR(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Subset of Regressors
        '''
        if self._fit_matrix is None or self._fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        ref_vectors = self._create_covMatrix(self.X_ref,X,self.kernel.eval)

        means = ref_vectors@solve(self._fit_matrix,self._fit_vector)

        if return_std:
            vars =  self.noise_var * np.array([k.T@solve(self._fit_matrix,k) for k in ref_vectors])
            stds = np.sqrt(vars)

            return means, stds
        
        return means
    
    def _predict_Ny(self, X, return_std=False):
        '''
            Calculates the posterior means and standard deviations for the Subset of Regressors
        '''
        if self._fit_matrix is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        data_vectors = self._create_covMatrix(X, self.X_data[:self.num_datapoints], self.kernel.eval)
        deriv_vectors = self._create_covMatrix(X, self.X_data[self.num_datapoints:], self.kernel.dx2)
        full_vectors = np.concatenate((data_vectors,deriv_vectors),axis=0).T
        # full_vectors = self._create_covMatrix(X, self.X_data, self.kernel.eval).T

        means = full_vectors@self._fit_matrix@self.Y_data

        if return_std:
            X_cov = self.kernel.eval(X,X)
            temp = np.array([k.T@self._fit_matrix@k for k in full_vectors])
            # stds = np.sqrt(X_cov + self.noise_var - temp)
            stds = np.sqrt(X_cov - temp) # no noise term in the variance

            return means, stds

        return means
        

    def _create_covMatrix(self, reference_points, data_points, kernel):
        xs,ys = np.meshgrid(reference_points,data_points,sparse=True)
        mat = kernel(xs,ys)
        
        return mat

class RBF_1D:
    def __init__(self,length_scale=1.0,coeff=1.0):
        self.length_scale = length_scale
        self.coeff = coeff

    def eval(self,x1,x2):
        return self.coeff*np.exp(-0.5/self.length_scale**2 * (x1-x2)**2)
    
    def dx2(self,x1,x2):
        return (x1-x2)/self.length_scale**2 * self.eval(x1,x2)
    
    def ddx1x2(self,x1,x2):
        return (1 - (x1-x2)**2 / self.length_scale**2) / self.length_scale**2 * self.eval(x1,x2)
    
class sparseGPR_ND:
    pass

class RBF_ND:
    def __init__(self,length_scale=1.0,coeff=1.0):
        self.length_scale = length_scale
        self.coeff = coeff

    def eval(self,x1,x2):
        diff = (x1-x2) / self.length_scale
        return self.coeff*np.exp(-0.5 * np.inner(diff,diff))
    
    def dx2(self,x1,x2):
        return (x1-x2)/self.length_scale**2 * self.eval(x1,x2)
    
    def ddx1x2(self,x1,x2):
        return (1 - (x1-x2)**2 / self.length_scale**2) / self.length_scale**2 * self.eval(x1,x2)