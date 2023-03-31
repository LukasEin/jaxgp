import numpy as np

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