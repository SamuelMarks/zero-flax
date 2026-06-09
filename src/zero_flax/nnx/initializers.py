"""Initializers."""

import numpy as np  # noqa: F401
from typing import Any, Tuple  # noqa: F401


def zeros(key: Any, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
    return np.zeros(shape, dtype=dtype)


def ones(key: Any, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
    return np.ones(shape, dtype=dtype)


def glorot_uniform(key: Any, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
    # Simplified
    limit = np.sqrt(6.0 / (shape[0] + shape[1])) if len(shape) >= 2 else 0.1
    # Mocking jax random with numpy random for eager execution tests
    if hasattr(key, "shape"):  # It's an array/tensor
        return np.random.uniform(-limit, limit, shape).astype(dtype)
    return np.zeros(shape, dtype=dtype)  # fallback


def he_normal(key: Any, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
    stddev = np.sqrt(2.0 / shape[0]) if len(shape) >= 1 else 0.1
    if hasattr(key, "shape"):
        return np.random.normal(0, stddev, shape).astype(dtype)
    return np.zeros(shape, dtype=dtype)
