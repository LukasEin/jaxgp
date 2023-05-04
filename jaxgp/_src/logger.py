import time
from typing import Tuple

import jax.numpy as jnp


class Logger:
    '''A simple logger to write out the convergence process of the optimization
    '''
    def __init__(self, name=None) -> None:
        if name is None:
            self.name = f"{time.time()}.log"
        else:
            self.name = f"{name}.log"

        self.buffer = []
        self.iters_list = []

    def __call__(self, output: Tuple) -> None:
        '''Appends the current parameters in the iteration to the buffer

        Parameters
        ----------
        output : Tuple
            current parameters of the optimization process
        '''
        self.buffer.append(output)

    def write(self):
        '''called after the optimization to save the parameters of the current optimization run
        '''
        self.iters_list.append(jnp.array(self.buffer))

        self.buffer = []

    def log(self, msg: str):
        '''log sum message

        Parameters
        ----------
        msg : str
            message to append to logfile
        '''
        with open(self.name, mode="a") as f:
            f.write(msg+"\n")
