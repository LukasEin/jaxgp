from typing import NamedTuple, Union

from jax.numpy import ndarray


class SparsePriorDistribution(NamedTuple):
    '''Prior distribution of a sparse GPR model.
    '''
    x_ref: ndarray
    U_ref: ndarray
    U_inv: ndarray
    diag: ndarray
    proj_labs: ndarray

class FullPriorDistribution(NamedTuple):
    '''Prior distribution of a full GPR model.
    '''
    x_data: ndarray
    y_data: ndarray
    k_nn: ndarray

class PosteriorDistribution(NamedTuple):
    '''Posterior distribution described by a mean and std vector'''
    mean: ndarray
    std: ndarray