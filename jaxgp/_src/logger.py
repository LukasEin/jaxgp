import jax.numpy as jnp
import time

class Logger:
    def __init__(self, name=None) -> None:
        if name is None:
            self.name = f"{time.time()}.log"
        else:
            self.name = f"{name}.log"

        self.buffer = []

    def __call__(self, output):
        with open(self.name, mode="a") as f:
            self.buffer.append(output)

    def write(self, loss):
        with open(self.name, mode="a") as f:
            f.write("# num_iter   params   loss\n")
            for i,param in enumerate(self.buffer):
                f.write(f"{i+1}   {param}   {loss(param)}\n")

            f.write("-"*30 + "\n\n")
        
        self.buffer = []