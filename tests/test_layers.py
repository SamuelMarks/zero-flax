"""Tests for zero_flax core layers."""

import numpy as np
from zero_flax.nnx import (
    Dense,
    Conv,
    Embed,
    LayerNorm,
    RMSNorm,
    MultiHeadAttention,
)


def test_dense():
    from jax.nn import initializers

    d = Dense(2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros)
    x = np.ones((1, 2))
    y = d(x)
    assert getattr(y, "shape", y.shape) == (1, 3)


def test_conv():
    c = Conv(3, 4, (3, 3))
    x = np.ones((1, 10, 10, 3))
    y = c(x)
    assert y.shape == (1, 10, 10, 4)


def test_embed():
    e = Embed(10, 4)
    x = np.array([[1, 2]])
    y = e(x)
    assert y.shape == (1, 2, 4)


def test_layernorm():
    ln = LayerNorm(4)
    x = np.ones((2, 4))
    y = ln(x)
    assert y.shape == (2, 4)


def test_rmsnorm():
    rn = RMSNorm(4)
    x = np.ones((2, 4))
    y = rn(x)
    assert y.shape == (2, 4)


def test_attention():
    attn = MultiHeadAttention(2, 4)
    x = np.ones((1, 5, 4))
    attn(x, x, x)


def test_dense_no_bias():
    d = Dense(2, 3, use_bias=False)
    x = np.ones((1, 2))
    y = d(x)
    assert y.shape == (1, 3)


def test_missing_apis():
    from zero_flax.nnx import missing

    axis = missing.AxisName()
    prec = missing.PrecisionLike()
    pad = missing.PaddingLike()
    dtype = missing.Dtype()
    shape = missing.Shape()
    axes = missing.Axes()
    size = missing.Size()
    ax = missing.Axis()
    dg = missing.DotGeneralT()
    mf = missing.MaxFun()

    assert axis is not None
    assert prec is not None
    assert pad is not None

    # Try dummy module
    import zero_flax.nnx as nnx
    from zero_flax.nnx.missing import variables

    var = variables.Variable()
    assert var is not None
