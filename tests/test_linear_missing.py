"""Unit tests for `Einsum`, `LinearGeneral`, and `Linear` initialization options."""

from __future__ import annotations

import numpy as np
import pytest
from zero_jax.nn import initializers

import zero_flax.nnx as nnx


def test_einsum() -> None:
    """Verify Einsum forward pass values, ellipsis equation normalization, and invalid equation error."""
    e = nnx.Einsum(
        "...ab,bc",
        (3, 4),
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    x = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    out = np.asarray(e(x))
    assert out.shape == (2, 4)
    np.testing.assert_allclose(
        out,
        np.array([[6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Invalid einsum equation"):
        nnx.Einsum("ab_no_comma", (2, 2))


def test_linear_general() -> None:
    """Verify LinearGeneral forward pass with tuple out_features and asymmetric inputs."""
    lg = nnx.LinearGeneral(
        3,
        (4, 5),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    x = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    out = np.asarray(lg(x))
    assert out.shape == (2, 4, 5)
    np.testing.assert_allclose(out[0], np.full((4, 5), 7.0, dtype=np.float32))
    np.testing.assert_allclose(out[1], np.full((4, 5), 3.0, dtype=np.float32))


def test_linear_init() -> None:
    """Verify Linear use_bias=True and use_bias=False configurations."""
    l_bias = nnx.Linear(
        3,
        4,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        use_bias=True,
    )
    l_no_bias = nnx.Linear(3, 4, use_bias=False, kernel_init=initializers.ones)
    assert l_bias.use_bias
    assert l_bias.bias is not None
    assert not l_no_bias.use_bias
    assert l_no_bias.bias is None
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(l_bias(x)), np.asarray(l_no_bias(x)))
