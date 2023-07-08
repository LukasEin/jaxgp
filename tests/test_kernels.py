import jax.numpy as jnp
import pytest

from calcgp.kernels import *

class TestBase:
    kernel = BaseKernel()
    X = jnp.zeros((5,))
    params = jnp.ones(2)

    def test_base_params(self):
        assert self.kernel.num_params == 2

    def test_base_eval(self):
        with pytest.raises(NotImplementedError):
            self.kernel.eval(self.X, self.X, self.params)

    def test_base_grad(self):
        with pytest.raises(NotImplementedError):
            self.kernel.grad2(self.X, self.X, self.params)

    def test_base_grad(self):
        with pytest.raises(NotImplementedError):
            self.kernel.jac(self.X, self.X, self.params)

class TestRBF:
    num_params = 2
    params = jnp.ones(num_params)
    kernel = RBF()
    X = jnp.zeros((5,))    

    def test_params(self):
        assert self.kernel.num_params == self.num_params

    def test_pointshape(self):
        XF = jnp.zeros((5,1))  

        # 1 wrong thing
        with pytest.raises(AssertionError): self.kernel.eval(XF, self.X, self.params)
        with pytest.raises(AssertionError): self.kernel.eval(self.X, XF, self.params)
        # 2 wrong things
        with pytest.raises(AssertionError): self.kernel.eval(XF, XF, self.params)

    def test_paramshape(self):
        params2 = jnp.ones((self.num_params,1)) 

        with pytest.raises(AssertionError): self.kernel.eval(self.X, self.X, params2)

    def test_returnshape(self):
        assert self.kernel.eval(self.X, self.X, self.params).shape == ()
        assert self.kernel.grad2(self.X, self.X, self.params).shape == (5,)
        assert self.kernel.jac(self.X, self.X, self.params).shape == (5,5)
    
class TestLinear(TestRBF):
    kernel = Linear()

class TestPeriodic(TestRBF):
    num_params = 3
    params = jnp.ones(num_params)
    kernel = Periodic()

class TestSum(TestRBF):
    num_params = 4
    params = jnp.ones(num_params)
    left_kernel = RBF(2)
    right_kernel = Linear(2)
    kernel = SumKernel(left_kernel, right_kernel)
   
class TestProduct(TestRBF):
    num_params = 4
    params = jnp.ones(num_params)
    left_kernel = RBF(2)
    right_kernel = Linear(2)
    kernel = ProductKernel(left_kernel, right_kernel)