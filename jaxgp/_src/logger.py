import jax.numpy as jnp
import time
from jax import vmap

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
        self.iters_list = []

        with open(self.name, mode="w") as f:
            pass

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
        if self.buffer:
            params = jnp.array(self.buffer)
            fun = vmap(loss, in_axes=0)
            losses = fun(params)

            self.iters_list.append((params, losses))

            with open(self.name, mode="a") as f:
                f.write("# num_iter   params   loss\n")
                for i,(param, loss) in enumerate(zip(params, losses)):
                    f.write(f"{i+1}   {param}   {loss}\n")

                f.write("-"*30 + "\n\n")
        
        else:
            with open(self.name, mode="a") as f:
                f.write("# num_iter   params   loss\n")
                f.write("-"*30 + "\n\n")

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
