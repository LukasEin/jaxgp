import jax.numpy as jnp
import time

from typing import Tuple, Callable

class Logger:
    '''A simple logger to write out the convergence process of the optimization
    '''
    def __init__(self, name=None) -> None:
        if name is None:
            self.name = f"{time.time()}.log"
        else:
            self.name = f"{name}.log"

        self.buffer = []

    def __call__(self, output: Tuple) -> None:
        '''Appends the current parameters in the iteration to the buffer

        Parameters
        ----------
        output : Tuple
            current parameters of the optimization process
        '''
        self.buffer.append(output)

    def write(self, loss: Callable):
        '''called after the optimization to write the parameters and 
        the corresponding loss values to a log file

        Parameters
        ----------
        loss : Callable
            loss function that takes the parameters in the buffer as input
        '''
        with open(self.name, mode="a") as f:
            f.write("# num_iter   params   loss\n")
            for i,param in enumerate(self.buffer):
                f.write(f"{i+1}   {param}   {loss(param)}\n")

            f.write("-"*30 + "\n\n")
        
        self.buffer = []