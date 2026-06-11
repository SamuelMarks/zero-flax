import pytest
import numpy as np
import zero_flax.nnx as nnx


def test_einsum():
    e = nnx.Einsum("ab,bc->ac", (3, 4))
    x = np.ones((2, 3))
    assert e(x).shape == (2, 4)


def test_linear_general():
    lg = nnx.LinearGeneral(3, (4, 5))
    x = np.ones((2, 3))
    assert lg(x).shape == (2, 4, 5)


def test_linear_init():
    from jax.nn import initializers

    l = nnx.Linear(
        3, 4, kernel_init=initializers.ones, bias_init=initializers.zeros, use_bias=True
    )
    l2 = nnx.Linear(3, 4, use_bias=False)
    assert l.use_bias
    assert not l2.use_bias
