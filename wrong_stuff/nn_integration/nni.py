import numpy as np

def integrate_1D_1L(bounds,weights,biases,integrated_activation_func):
    '''
    weights.shape = (2,dim_1,dim_2) 
        ... dim_1,dim_2 dimensions of each weight matrix
        ... 2 = number of hidden layers + 1
    '''
    weight1 = weights[0].reshape(1,-1)
    weight1 = weight1[0]
    weight2 = weights[1].reshape(1,-1)
    weight2 = weight2[0]

    lowerbounds = bounds[0]*weight1 + biases[0]
    upperbounds = bounds[1]*weight1 + biases[0]

    integral = np.sum(weight2 / weight1*(integrated_activation_func(upperbounds) - integrated_activation_func(lowerbounds))) + (bounds[1] - bounds[0]) *biases[1][0]

    return integral

def integrate_ND_1L(bounds,weights,biases,act):
    '''
    bounds  ... iterable with 2 elements each being an N-dimensional point. 
                Integration is performed from bounds[0] to bounds[1].
    weights ... iterable with 2 weight-matrices. Dims must be NxM for first matrix and MxN for the second. 
                M is given by the number of hidden layers, N must be the same as the dimensions of the bound points
    biases  ... iterable with 2 bias-vectors, first must be M-dimensional according to weights,
                second one must be N dimensional according to bounds
    act     ... type of activation function that was used. 
                !!! Must be supported in this file below !!!
    '''
    assert act in ["tanh", "relu"], "Activation function given is not supported"

    if act == "tanh":
        activation_func = tanh
        integrated_activation_func = int_tanh
    elif act == "relu":
        activation_func = relu
        integrated_activation_func = int_relu

    weight1 = weights[0]
    weight2 = weights[1]

    lowerbounds = weight1.T@bounds[0] + biases[0]
    upperbounds = weight1.T@bounds[1] + biases[0]
    divisors = upperbounds - lowerbounds

    integral_vector = np.zeros_like(divisors)

    integral_vector[divisors != 0] = \
        (integrated_activation_func(upperbounds[divisors != 0]) - integrated_activation_func(lowerbounds[divisors != 0])) / divisors[divisors != 0]
    
    integral_vector[divisors == 0] = activation_func(lowerbounds[divisors==0])

    return (bounds[1] - bounds[0]).T@weight2.T@integral_vector + (bounds[1] - bounds[0]).T@biases[1]

# Varioous activation functions and their integrals
def relu(x):
    return np.maximum(x,0)

def int_relu(x):
    return np.maximum(np.sign(x)*x**2,0)

def tanh(x):
    return np.tanh(x)

def int_tanh(x):
    return np.log(np.cosh(x))





