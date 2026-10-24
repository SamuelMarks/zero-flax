"""Unit tests for `Linear`, `LinearGeneral`, `Einsum`, `Dropout`, and `BatchNorm`."""

from __future__ import annotations

import numpy as np

from zero_flax.nnx.linear import Einsum, Linear, LinearGeneral
from zero_flax.nnx.normalization import BatchNorm
from zero_flax.nnx.stochastic import Dropout


def test_linear() -> None:
    """Verify Linear, LinearGeneral, and Einsum forward pass shapes."""
    x = np.ones((1, 2), dtype=np.float32)
    lin = Linear(2, 3)
    y = lin(x)
    assert y.shape == (1, 3)

    lg = LinearGeneral(2, 3)
    y2 = lg(x)
    assert y2.shape == (1, 3)

    e = Einsum("ab,bc->ac", (2, 3), (3,))
    y3 = e(x)
    assert y3.shape == (1, 3)


def test_dropout() -> None:
    """Verify Dropout in stochastic and deterministic modes."""
    d = Dropout(0.5)
    x = np.ones((2, 2), dtype=np.float32)
    y = d(x)
    assert y.shape == x.shape
    y2 = d(x, deterministic=True)
    np.testing.assert_allclose(np.asarray(y2), x)


def test_batch_norm() -> None:
    """Verify BatchNorm forward pass with use_running_average False and True."""
    b = BatchNorm(2)
    x = np.ones((2, 2), dtype=np.float32)
    y = b(x)
    assert y.shape == x.shape
    y2 = b(x, use_running_average=True)
    assert y2.shape == x.shape
