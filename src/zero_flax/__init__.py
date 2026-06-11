from __future__ import annotations

"""Zero-Flax root package initialization.

This package provides a lightweight compatibility layer and implementations
for neural network components inspired by or compatible with Flax NNX.
"""

import os

os.environ["SWITCHEROO_EAGER_MODE"] = "1"

__version__ = "0.8.5"
from . import nnx

__all__ = ["nnx"]
