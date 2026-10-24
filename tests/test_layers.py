"""Tests for core layers (`Dense`, `Conv`, `Embed`, `LayerNorm`, `RMSNorm`, `MultiHeadAttention`) and activation functions."""

from __future__ import annotations

import numpy as np
from zero_jax.nn import initializers

import zero_flax.nnx as nnx
from zero_flax.nnx import layers, variables


def test_dense_forward() -> None:
    """Verify Dense forward pass with explicit ones/zeros initializers."""
    d = layers.Dense(2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros)
    x = np.ones((1, 2), dtype=np.float32)
    y = d(x)
    assert y.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(y), np.full((1, 3), 2.0, dtype=np.float32))


def test_dense_no_bias() -> None:
    """Verify Dense without bias initializes bias=None and computes x @ W."""
    d = layers.Dense(2, 3, use_bias=False, kernel_init=initializers.ones)
    assert d.bias is None
    x = np.ones((1, 2), dtype=np.float32)
    y = d(x)
    assert y.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(y), np.full((1, 3), 2.0, dtype=np.float32))


def test_conv() -> None:
    """Verify Conv 2D forward pass with integer and tuple kernel_size."""
    c = layers.Conv(
        2,
        3,
        kernel_size=(3, 3),
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    x = np.ones((1, 4, 4, 2), dtype=np.float32)
    y = c(x)
    assert y.shape == (1, 4, 4, 3)
    assert float(np.asarray(y)[0, 1, 1, 0]) == 18.0

    c2 = layers.Conv(
        2,
        3,
        kernel_size=3,
        use_bias=False,
        kernel_init=initializers.ones,
    )
    assert c2.bias is None
    x_1d = np.ones((1, 4, 2), dtype=np.float32)
    y2 = c2(x_1d)
    assert y2.shape == (1, 4, 3)
    assert float(np.asarray(y2)[0, 1, 0]) == 6.0


def test_embed() -> None:
    """Verify Embed token lookup and attend logit projection."""
    e = layers.Embed(10, 4, embedding_init=initializers.ones)
    x = np.array([1, 2, 3], dtype=np.int32)
    y = e(x)
    assert y.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(y), np.ones((3, 4), dtype=np.float32))

    q = np.ones((2, 4), dtype=np.float32)
    scores = e.attend(q)
    assert scores.shape == (2, 10)
    np.testing.assert_allclose(np.asarray(scores), np.full((2, 10), 4.0, dtype=np.float32))


def test_layernorm() -> None:
    """Verify LayerNorm with and without scale and bias."""
    ln = nnx.LayerNorm(4)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    y = np.asarray(ln(x))
    assert y.shape == (2, 4)
    np.testing.assert_allclose(np.mean(y, axis=-1), np.zeros((2,)), atol=1e-5)
    np.testing.assert_allclose(np.var(y, axis=-1), np.ones((2,)), atol=1e-4)

    ln2 = nnx.LayerNorm(4, use_bias=False, use_scale=False)
    assert ln2.bias is None
    assert ln2.scale is None
    y2 = np.asarray(ln2(x))
    np.testing.assert_allclose(y2, y, rtol=1e-5, atol=1e-5)


def test_rmsnorm() -> None:
    """Verify RMSNorm with and without scale."""
    rn = nnx.RMSNorm(4, epsilon=0.0)
    x = np.full((2, 4), 3.0, dtype=np.float32)
    y = np.asarray(rn(x))
    assert y.shape == (2, 4)
    np.testing.assert_allclose(y, np.ones((2, 4), dtype=np.float32), rtol=1e-5)

    rn2 = nnx.RMSNorm(4, epsilon=0.0, use_scale=False)
    assert rn2.scale is None
    y2 = np.asarray(rn2(x))
    np.testing.assert_allclose(y2, np.ones((2, 4), dtype=np.float32), rtol=1e-5)


def test_mha() -> None:
    """Verify MultiHeadDotProductAttention with self-attention and cross-attention."""
    mha = layers.MultiHeadDotProductAttention(
        num_heads=2,
        in_features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    q = np.ones((1, 3, 4), dtype=np.float32)
    k = np.ones((1, 3, 4), dtype=np.float32)
    v = np.ones((1, 3, 4), dtype=np.float32)
    y = mha(q, k, v)
    assert y.shape == (1, 3, 4)
    np.testing.assert_allclose(np.asarray(y), np.full((1, 3, 4), 16.0, dtype=np.float32), rtol=1e-5)

    y_self = mha(q)
    np.testing.assert_allclose(
        np.asarray(y_self), np.full((1, 3, 4), 16.0, dtype=np.float32), rtol=1e-5
    )

    mha2 = layers.MultiHeadDotProductAttention(
        num_heads=2,
        in_features=4,
        qkv_features=4,
        out_features=2,
        use_bias=False,
        kernel_init=initializers.ones,
    )
    y2 = mha2(q, k)
    assert y2.shape == (1, 3, 2)
    np.testing.assert_allclose(
        np.asarray(y2), np.full((1, 3, 2), 16.0, dtype=np.float32), rtol=1e-5
    )


def test_nn_activations_reexport() -> None:
    """Verify all activations and helpers re-exported on zero_flax.nnx match analytical values."""
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)

    np.testing.assert_allclose(
        np.asarray(nnx.relu(x)), np.array([0.0, 0.0, 2.0], dtype=np.float32), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.relu6(np.array([-1.0, 3.0, 8.0], dtype=np.float32))),
        np.array([0.0, 3.0, 6.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.sigmoid(np.array([0.0], dtype=np.float32))),
        np.array([0.5], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.soft_sign(x)), x / (1.0 + np.abs(x)), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.softplus(x)), np.logaddexp(x, 0.0), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.leaky_relu(x, negative_slope=0.1)),
        np.where(x >= 0, x, 0.1 * x),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.elu(x)), np.where(x > 0, x, np.exp(x) - 1.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.celu(x)), np.maximum(0.0, x) + np.minimum(0.0, np.exp(x) - 1.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.silu(x)), x / (1.0 + np.exp(-x)), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.swish(x)), np.asarray(nnx.silu(x)), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx.hard_tanh(np.array([-2.0, 0.5, 2.0], dtype=np.float32))),
        np.array([-1.0, 0.5, 1.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.hard_sigmoid(np.array([0.0], dtype=np.float32))),
        np.array([0.5], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.hard_swish(np.array([0.0, 3.0], dtype=np.float32))),
        np.array([0.0, 3.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.hard_silu(np.array([0.0, 3.0], dtype=np.float32))),
        np.array([0.0, 3.0], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.gelu(np.array([0.0], dtype=np.float32))),
        np.array([0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.selu(np.array([0.0], dtype=np.float32))),
        np.array([0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(nnx.log_sigmoid(np.array([0.0], dtype=np.float32))),
        np.array([-np.log(2.0)], dtype=np.float32),
        rtol=1e-5,
    )
    sm = np.asarray(nnx.softmax(np.array([1.0, 1.0], dtype=np.float32)))
    np.testing.assert_allclose(sm, np.array([0.5, 0.5], dtype=np.float32), rtol=1e-5)
    lsm = np.asarray(nnx.log_softmax(np.array([1.0, 1.0], dtype=np.float32)))
    np.testing.assert_allclose(lsm, np.log(sm), rtol=1e-5)

    glu_out = np.asarray(nnx.glu(np.zeros((2, 4), dtype=np.float32), axis=-1))
    assert glu_out.shape == (2, 2)
    np.testing.assert_allclose(glu_out, np.zeros((2, 2), dtype=np.float32), atol=1e-6)

    lse = float(np.asarray(nnx.logsumexp(np.array([0.0, 0.0], dtype=np.float32))))
    np.testing.assert_allclose(lse, np.log(2.0), rtol=1e-5)

    oh = np.asarray(nnx.one_hot(np.array([0, 2]), 3))
    np.testing.assert_allclose(oh, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))

    std = np.asarray(nnx.standardize(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)))
    np.testing.assert_allclose(np.mean(std), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.var(std), 1.0, atol=1e-4)

    init = nnx.Initializer()
    assert init((0, 0), (2, 2)).shape == (2, 2)

    dg = nnx.DotGeneralT()
    assert dg(x, x, (((0,), (0,)), ((), ()))) is not None

    mf = nnx.MaxFun()
    np.testing.assert_allclose(
        np.asarray(mf(x, np.zeros(3, dtype=np.float32))),
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
    )

    dt = nnx.Dtype("float32")
    assert str(dt) == "float32"

    sh = nnx.Shape((2, 3))
    assert tuple(sh) == (2, 3)

    ax = nnx.Axis(0)
    assert int(ax) == 0

    axes = nnx.Axes((0, 1))
    assert tuple(axes) == (0, 1)

    sz = nnx.Size(10)
    assert int(sz) == 10

    ax_name = nnx.AxisName("batch")
    assert str(ax_name) == "batch"

    prec = nnx.PrecisionLike("high")
    assert prec.precision == "high"

    pad = nnx.PaddingLike("SAME")
    assert pad.padding == "SAME"

    var = variables.Variable(5)
    assert var.value == 5
