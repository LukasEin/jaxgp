import jax.numpy as jnp
import pytest

from jaxgp.kernels import *

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
    kernel = RBF()
    params = jnp.ones(2)
    X = jnp.zeros((5,))    

    def test_params(self):
        print(self.kernel)
        assert self.kernel.num_params == 2

    def test_pointshape(self):
        XF = jnp.zeros((5,1))  

        # 1 wrong thing
        with pytest.raises(AssertionError): self.kernel.eval(XF, self.X, self.params)
        with pytest.raises(AssertionError): self.kernel.eval(self.X, XF, self.params)
        # 2 wrong things
        with pytest.raises(AssertionError): self.kernel.eval(XF, XF, self.params)

    def test_paramshape(self):
        params2 = jnp.ones((2,2)) 

        with pytest.raises(AssertionError): self.kernel.eval(self.X, self.X, params2)

    def test_returnshape(self):
        assert self.kernel.eval(self.X, self.X, self.params).shape == ()
        assert self.kernel.grad2(self.X, self.X, self.params).shape == (5,)
        assert self.kernel.jac(self.X, self.X, self.params).shape == (5,5)
    
class TestLinear(TestRBF):
    kernel = Linear()

class TestPeriodic(TestRBF):
    kernel = Periodic()