# JaxGP: Gaussian Process Regression in JAX

## Introduction
JaxGP is a Gaussian Process Regression framework based on [JAX](https://github.com/google/jax). 

The key difference to other GPR frameworks is the flexibility in choosing different datapoints for function and gradient observations. The idea behind this is to mainly use gradient observations and thereby formally integrate the gradient to recover the true function.

## Examples

`1d_example_regression.ipynb` and `2d_example_regression.ipynb` show hands on examples on how to use the framework to recover a scalar valued function $f: \mathbb{R}^d \to \mathbb{R}$ from observing its gradient $\nabla f: \mathbb{R}^d \to \mathbb{R}^d$.

## Notes testing

- Only the `eval` methods in the pure (non-combined) kernels need `asserts` for the shape. The gradients and "hessians" as well as any combined kernel `eval` methods call functions that already have asserts.