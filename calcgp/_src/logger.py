from typing import Tuple


class Logger:
    '''A simple logger to write out the convergence process of the optimization
    '''
    def __init__(self) -> None:
        self.iters_list = []

    def __call__(self, output: Tuple) -> None:
        '''Appends the current parameters in the iteration to the buffer

        Parameters
        ----------
        output : Tuple
            current parameters of the optimization process
        '''
        self.iters_list.append(output)