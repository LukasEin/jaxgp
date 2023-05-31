# Errors where?
This document is supposed to go through all possible sources of errors to find out why the sparse model does not work.

## :white_check_mark: The covariance matrices are built wrong 

This was tested in the notebook `compare_covar_pytorch.ipynb`. All relevant covariance matrices (`K_NN, K_MN, K_MM`) where compared between my [JAX](https://github.com/google/jax) implementation and the package [gpytorch](https://gpytorch.ai/). All three matrices are the same up to a numerical error of oreder $\mathcal{O}(10^{-6})$. No noise term added to compare the true covariance matrices.

## :white_check_mark: The covariance matrices are not positive semidefinite

This was also tested in the notebook `compare_covar_pytorch.ipynb`. In this case one has to add the noise term to the `K_NN` matrix since it is only inverted in the form `K_NN + Id*noise**2`. The `K_MM` matrix is inverted as is and no diagonal addition should be needed to make the matrix positive definite.

In some cases `K_NN` itself has small negative eigenvalues however an additional noise term makes them all larger than zero.

*Note: For stability of the inversion it is helpful to a a small diagonal element to all matrices that are supposed to be inverted. The should not influence the prediction too much but makes the inversion more stable.*

## :white_check_mark: The input data is shaped/ordered wrong

Each row of the covariance matrices describes the covariance between **one** point with all others. Assuming that the datapoints are all in $\mathbb{R}^2$, a row of the covariance matrix looks as follows:
 
$$
\begin{align}
    \bigg[k(x, x_1^f), ..., k(x, x_n^f), \frac{\partial}{\partial x_{1,1}^d}k(x, x_1^d), \frac{\partial}{\partial x_{1,2}^d}k(x, x_1^d), ..., \frac{\partial}{\partial x_{m,1}^d}k(x, x_m^d), \frac{\partial}{\partial x_{m,2}^d}k(x, x_m^d)\bigg].
\end{align}
$$

I.e., first the covariance between $x$ and all points at which the function was evaluated is calculated and then for each point where the gradient is known, all partials of the kernel are calculated.

The rows are ordered in the same way.

If the inputs for the training are
```python
def f(X):
    return function(X)

def df(X):
    dx1 = grad_function_x1(X)
    ...
    dxn = grad_function_xN(X)
    return jnp.vstack((dx1, ..., dx1N)).T

X_train = (X_func, X_grad)
Y_train = jnp.vstack(f(X_func), df(X_grad).reshape(-1)), 
```

then all the inputs are given the way they are supposed to. `Y_train` has the same ordering as the rows/columns of the covariance matrix. This can be seen in the notebook `2d_integration_example.ipynb`.

## :white_check_mark: Put the matrices together correctly

For the full GPR model there is nothing to do here. The matrix is already as it should be. For details look at my Latex file here I only put the formulas of interest.

For calculating the inverse of the covariance matrix we use:
$$
\begin{align}
    \Sigma_{FITC}(\mathbf{x}, \mathbf{x})^{-1} &= \Lambda^{-1}(\mathbf{x}, \mathbf{x}) - \Lambda^{-1}(\mathbf{x}, \mathbf{x})K(\mathbf{x}, \mathbf{x}_I)B_I(\mathbf{x}, \mathbf{x})^{-1}K(\mathbf{x}_I, \mathbf{x})\Lambda^{-1}(\mathbf{x}, \mathbf{x}) \\
    B_I(\mathbf{x}, \mathbf{x}) &= K(\mathbf{x}_I, \mathbf{x}_I) + K(\mathbf{x}_I, \mathbf{x})\Lambda^{-1}(\mathbf{x}, \mathbf{x})K(\mathbf{x}, \mathbf{x}_I) \\
    \Lambda(\mathbf{x}, \mathbf{x}) &= \text{diag}\big(K(\mathbf{x}, \mathbf{x}) - K(\mathbf{x}, \mathbf{x}_I)K^{-1}(\mathbf{x}_I, \mathbf{x}_I)K(\mathbf{x}_I, \mathbf{x})\big) + \mathbf{I}\sigma^2
\end{align}
$$
where $\Sigma_{FITC}$ is the approximate covariance matrix.

The function `sparse_covariance_matrix` calculates the following:
- :white_check_mark: $\Lambda(\mathbf{x}, \mathbf{x})$. Since it is a diagonal matrix stored simply as a vector and it is easy to invert by just inverting element-wise. It is also easy to calculate the MVM between $\Lambda$ and an arbitrary vector by just doing element wise division. In `fitc_diagonal.ipynb` the implementation of this matrix was tested.

- :white_check_mark: $K(\mathbf{x}_I, \mathbf{x}_I):=$`K_MM` which was already checked to be correct in `compare_covar_pytorch.ipynb`

- :white_check_mark: $B_I(\mathbf{x}, \mathbf{x})$. Since $\Lambda(\mathbf{x}, \mathbf{x})$, `K_MM` and $K(\mathbf{x}_I, \mathbf{x}):=$`K_MN` have already been checked as correct it is only necessary to check if the combination of those matrices follows the mathematical equation. The code to calculate it looks as follows, which has the same form as equation (3): `K_inv = K_ref + K_MN@jnp.diag(1 / fitc_diag)@K_MN.T`
:heavy_exclamation_mark: This was implemented incorrectly since `K_ref` was already cholesky transformed before its use in the equation before :heavy_exclamation_mark:
- :white_check_mark: $K(\mathbf{x}_I, \mathbf{x})\Lambda^{-1}(\mathbf{x}, \mathbf{x})Y$ the labels $Y$ projected into the reference space. Again since all elements of this equation are already checked to be correct, the only thing left is to see if the calculation is correctly implemented: `projected_label = K_MN@(Y_data / fitc_diag)` which is exactly how it is supposed to be. The MVM between the diagonal matrix and the data vector is done via element-wise division.

## :x: Correctly calculating the log marginal likelihood

When calculating the log marginal likelihood it most of the time return a `nan` value. To see where this comes from, I let the function return all intermediate results. The culprit here is `K_inv` or $B_I(\mathbf{x}, \mathbf{x})$ from the previous point. Even though it is implemented correctly and I even added a small diagonal addition to it of $10^{-4}$ the **Cholesky factorization** returns a decomposition that is an upper triagular matrix with only nan values on in the upper triangular part.

The maximum elements of `K_inv` become hugh, maybe divide by large number first lets see.