"""Unit tests for `MultiHeadAttention` and `MultiHeadDotProductAttention`."""

from __future__ import annotations

import numpy as np
import pytest
from zero_jax.nn import initializers

import zero_flax.nnx as nnx
from zero_flax.nnx.layers import MultiHeadDotProductAttention


def test_multihead() -> None:
    """Verify MultiHeadAttention forward pass and input validation."""
    m = nnx.MultiHeadAttention(2, 4, kernel_init=initializers.ones, bias_init=initializers.zeros)
    with pytest.raises(ValueError, match="non-None inputs_q"):
        m(None, None, None)

    q = np.ones((2, 4), dtype=np.float32)
    k = np.ones((2, 4), dtype=np.float32)
    v = np.ones((2, 4), dtype=np.float32)
    out = m(q, k, v)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(np.asarray(out), np.full((2, 4), 16.0, dtype=np.float32))


def test_multihead2() -> None:
    """Verify MultiHeadDotProductAttention handles 3D sequence inputs."""
    m = MultiHeadDotProductAttention(
        2, 4, kernel_init=initializers.ones, bias_init=initializers.zeros
    )
    q = np.ones((1, 2, 4), dtype=np.float32)
    k = np.ones((1, 2, 4), dtype=np.float32)
    v = np.ones((1, 2, 4), dtype=np.float32)
    out = m(q, k, v)
    assert out.shape == (1, 2, 4)
    np.testing.assert_allclose(
        np.asarray(out), np.full((1, 2, 4), 16.0, dtype=np.float32), rtol=1e-5
    )


def test_multihead3() -> None:
    """Verify MultiHeadDotProductAttention handles 1D inputs and rejects invalid shapes."""
    m = MultiHeadDotProductAttention(
        2, 4, kernel_init=initializers.ones, bias_init=initializers.zeros
    )
    out_1d = m(np.ones((4,), dtype=np.float32))
    assert out_1d.shape == (4,)
    np.testing.assert_allclose(np.asarray(out_1d), np.full((4,), 16.0, dtype=np.float32))

    with pytest.raises(ValueError, match="Expected inputs_q"):
        m(1, 2, 3)
    with pytest.raises(ValueError, match="Expected inputs_k"):
        m(np.ones((2, 4), dtype=np.float32), np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected inputs_v"):
        m(
            np.ones((2, 4), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
        )


def test_multihead_decode_and_sow_weights() -> None:
    """Verify MultiHeadAttention autoregressive decode KV-cache and sow_weights recording."""
    m = nnx.MultiHeadAttention(
        num_heads=2,
        in_features=4,
        dropout_rate=0.1,
        deterministic=False,
        keep_rngs=True,
        rngs=nnx.Rngs(0),
    )
    x_step = np.ones((1, 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="init_cache"):
        m(x_step, decode=True)

    m.init_cache((1, 3, 4))
    assert m.cached_key is not None
    assert m.cached_value is not None
    assert m.cache_index is not None
    assert m.cache_index.value == 0

    out_step0 = m(x_step, decode=True, sow_weights=True, deterministic=False)
    assert out_step0.shape == (1, 1, 4)
    assert m.cache_index.value == 1
    assert m.attention_weights is not None
    assert m.attention_weights.value.shape == (1, 2, 1, 3)

    out_step1 = m(x_step, decode=True, deterministic=True)
    assert out_step1.shape == (1, 1, 4)
    assert m.cache_index.value == 2

    out_step2 = m(
        x_step,
        decode=True,
        deterministic=True,
        mask=np.ones((1, 1, 1, 3), dtype=bool),
    )
    assert out_step2.shape == (1, 1, 4)
    assert m.cache_index.value == 3
