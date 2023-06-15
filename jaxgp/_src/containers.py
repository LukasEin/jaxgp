from typing import NamedTuple

from jax.numpy import ndarray


class SparsePrior(NamedTuple):
    '''Prior distribution of a sparse GPR model.
    '''
    U_ref: ndarray
    U_inv: ndarray
    diag: ndarray
    proj_labs: ndarray

class FullPrior(NamedTuple):
    '''Prior distribution of a full GPR model.
    '''
    k_nn: ndarray
    y_data: ndarray

class Posterior(NamedTuple):
    '''Posterior distribution described by a mean and std vector'''
    mean: ndarray
    std: ndarray