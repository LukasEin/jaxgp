import numpy as np
from scipy import linalg, stats

def gausskernel(p1,p2,*,param=1.0):
    """Only for a function in 1D atm"""
    return np.exp(-(p1-p2)**2 / (2*param**2))

def cov_matrix(points_vec,*,param=1.0,alpha=1e-6):
    mat = np.zeros((len(points_vec),len(points_vec)))

    for i,p1 in enumerate(points_vec):
        for j,p2 in enumerate(points_vec):
            mat[i,j] = gausskernel(p1,p2,param=param)

    mat += np.eye(len(points_vec))*alpha

    return mat

def mean(numpoints,value):
    return np.ones(numpoints)*value

def post_mean(p_new,points_vec,func_vec,cov_matrix,param):
    cov_vec = gausskernel(p_new,points_vec,param)
    temp = linalg.solve(cov_matrix,func_vec)

    return cov_vec@temp

def post_var(p_new,points_vec,cov_matrix,param):
    cov_vec = gausskernel(p_new,points_vec,param)
    temp = linalg.solve(cov_matrix,cov_vec)

    return gausskernel(p_new,p_new,param) - cov_vec@temp

# def expected_improvement(post_mean,post_var,curr_max,expl_param):
#     if np.isclose(0,post_var,1e-15):
#         return 0
    
#     rv = stats.norm()

#     temp = post_mean - curr_max - expl_param
#     return temp * rv.cdf(temp / post_var) + post_var * rv.pdf(temp / post_var)

def expected_improvement(post_mean,post_var,curr_max,expl_param):
    if np.isclose(post_var,0,1e-10):
        return 0
    
    rv = stats.norm()
    temp = post_mean - curr_max - expl_param

    return max(temp,0) + post_var * rv.pdf(temp / post_var) - np.abs(temp) * rv.cdf(temp / post_var)