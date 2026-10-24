"""Unit tests for `Einsum`, `LinearGeneral`, and `Linear` initialization options."""

from __future__ import annotations

import numpy as np
from zero_jax.nn import initializers

import zero_flax.nnx as nnx


def test_einsum() -> None:
    """Verify Einsum forward pass shape and bias-free execution."""
    e = nnx.Einsum("ab,bc->ac", (3, 4))
    x = np.ones((2, 3), dtype=np.float32)
    assert e(x).shape == (2, 4)


def test_linear_general() -> None:
    """Verify LinearGeneral forward pass with tuple out_features."""
    lg = nnx.LinearGeneral(3, (4, 5))
    x = np.ones((2, 3), dtype=np.float32)
    assert lg(x).shape == (2, 4, 5)


def test_linear_init() -> None:
    """Verify Linear use_bias=True and use_bias=False configurations."""
    l_bias = nnx.Linear(
        3,
        4,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        use_bias=True,
    )
    l_no_bias = nnx.Linear(3, 4, use_bias=False)
    assert l_bias.use_bias
    assert l_bias.bias is not None
    assert not l_no_bias.use_bias
    assert l_no_bias.bias is None
