import numpy as np
from scipy.linalg import solve

class sparseGPR:
    def __init__(self,kernel,num_datapoints=None,num_derivpoints=None,X_ref=None,noise=1e-5) -> None:
        self.kernel = kernel

        self.num_datapoints = num_datapoints
        self.num_derivpoints = num_derivpoints

        # X_ref must be given for now!!!!!
        if X_ref is None:
            raise ValueError("X_ref can't be None!")
        self.X_ref = X_ref
        self.num_referencepoints = len(X_ref)
        self.K_MM = self._create_covMatrix(self.X_ref,self.X_ref,self.kernel.eval)

        self.noise = noise
        self.regress_coeff = None
        self.fit_vector = None
        self.fit_matrix = None

    def fit(self,X_data,Y_data) -> None:
        '''
            X_data ... (datapoint_1, ..., datapoint_N1, derivpoint_1, ..., derivpoint_N2)
            Y_data ... same shape as X_data
            
            for now X_ref must be given!!!!
        '''
        X_func = X_data[:self.num_datapoints]
        X_deriv = X_data[self.num_datapoints:]
        
        K_func = self._create_covMatrix(self.X_ref,X_func,self.kernel.eval)
        K_deriv = self._create_covMatrix(self.X_ref,X_deriv,self.kernel.dx2)
        K_MN = np.concatenate((K_func,K_deriv),axis=0)

        print(K_func.shape,K_deriv.shape,K_MN.shape,sep="\n\n")

        self.fit_matrix = self.K_MM + K_MN.T@K_MN / self.noise**2
        self.fit_vector = K_MN.T@Y_data / self.noise**2

    def predict(self,X, return_std=False):
        '''
            for now X_ref must be given in self.fit!!!
        '''
        if self.fit_matrix is None or self.fit_vector is None:
            raise ValueError("GPR not correctly fitted!")
        if self.X_ref is None:
            raise ValueError("X_ref can't be None!")
        
        ref_vectors = self._create_covMatrix(self.X_ref,X,self.kernel.eval)
        X_cov = self.kernel.eval(X,X)

        means = ref_vectors.T@solve(self.fit_matrix,self.fit_vector)

        if not return_std:
            stds = X_cov + self.noise**2 - ref_vectors.T@solve(self.K_MM + self.fit_matrix,ref_vectors)
            return means, stds
        
        return means
    
    def _create_covMatrix(self, reference_points, data_points, kernel):
        xs,ys = np.meshgrid(reference_points,data_points,sparse=True)
        mat = kernel(xs,ys)
        
        return mat

class RBF:
    def __init__(self,length_scale=1.0,coeff=1.0):
        self.length_scale = length_scale
        self.coeff = coeff

    def eval(self,x1,x2):
        return self.coeff*np.exp(-0.5/self.length_scale**2 * (x1-x2)**2)
    
    def dx2(self,x1,x2):
        return (x1-x2)/self.length_scale**2 * self.eval(x1,x2)
    
    def ddx1x2(self,x1,x2):
        return (1 - (x1-x2)**2 / self.length_scale**2) / self.length_scale**2 * self.eval(x1,x2)