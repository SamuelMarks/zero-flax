"""Unit tests for `Linear`, `LinearGeneral`, `Einsum`, `Dropout`, and `BatchNorm`."""

from __future__ import annotations

import numpy as np
from zero_jax.nn import initializers

from zero_flax.nnx.linear import Einsum, Linear, LinearGeneral
from zero_flax.nnx.normalization import BatchNorm
from zero_flax.nnx.stochastic import Dropout


def test_linear() -> None:
    """Verify Linear, LinearGeneral, and Einsum forward pass numerical values."""
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    lin = Linear(2, 3, kernel_init=initializers.ones, bias_init=initializers.ones)
    y = np.asarray(lin(x))
    assert y.shape == (1, 3)
    np.testing.assert_allclose(y, np.full((1, 3), 6.0, dtype=np.float32))

    lg = LinearGeneral(2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros)
    y2 = np.asarray(lg(x))
    assert y2.shape == (1, 3)
    np.testing.assert_allclose(y2, np.full((1, 3), 5.0, dtype=np.float32))

    e = Einsum(
        "ab,bc->ac",
        (2, 3),
        (3,),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y3 = np.asarray(e(x))
    assert y3.shape == (1, 3)
    np.testing.assert_allclose(y3, np.full((1, 3), 6.0, dtype=np.float32))


def test_dropout() -> None:
    """Verify Dropout in stochastic and deterministic modes."""
    d = Dropout(0.5)
    x = np.ones((2, 2), dtype=np.float32)
    y = np.asarray(d(x))
    assert y.shape == x.shape
    assert np.all((y == 0.0) | (y == 2.0))
    y2 = d(x, deterministic=True)
    np.testing.assert_allclose(np.asarray(y2), x)


def test_batch_norm() -> None:
    """Verify BatchNorm forward pass with use_running_average False and True."""
    b = BatchNorm(2, epsilon=0.0)
    x = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    y = np.asarray(b(x))
    assert y.shape == x.shape
    np.testing.assert_allclose(y, np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32), atol=1e-5)
    y2 = np.asarray(b(x, use_running_average=True))
    assert y2.shape == x.shape
