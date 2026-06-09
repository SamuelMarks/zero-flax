import numpy as np
from zero_flax.nnx.linear import Linear, LinearGeneral, Einsum
from zero_flax.nnx.stochastic import Dropout
from zero_flax.nnx.normalization import BatchNorm


def test_linear():
    x = np.ones((1, 2))
    lin = Linear(
        2, 3, kernel_init=lambda k, s: np.ones(s), bias_init=lambda k, s: np.zeros(s)
    )
    y = lin(x)
    assert y.shape == (1, 3)

    lg = LinearGeneral(
        2, 3, kernel_init=lambda k, s: np.ones(s), bias_init=lambda k, s: np.zeros(s)
    )
    y = lg(x)
    assert y.shape == (1, 3)

    e = Einsum(
        "ij,jk->ik",
        (2, 3),
        bias_shape=(3,),
        kernel_init=lambda k, s: np.ones(s),
        bias_init=lambda k, s: np.zeros(s),
    )
    y = e(x)
    assert y.shape == (1, 3)


def test_dropout():
    d = Dropout(0.5)
    x = np.ones((1, 2))
    y = d(x)
    assert y.shape == x.shape
    y2 = d(x, deterministic=True)
    assert np.all(y2 == x)


def test_batch_norm():
    b = BatchNorm(2)
    x = np.ones((1, 2))
    y = b(x)
    assert y.shape == x.shape
    y2 = b(x, use_running_average=True)
    assert y2.shape == x.shape
