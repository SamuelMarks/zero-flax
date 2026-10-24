"""Zero-Flax root package initialization.

This package provides a pure-Python, zero-framework-dependency implementation
of the Flax NNX neural network library backed by `ml-switcheroo-ir` and
`ml-switcheroo-compiler`.
"""

from __future__ import annotations

import os

os.environ.setdefault("SWITCHEROO_EAGER_MODE", "1")

from zero_flax import nnx as nnx
from zero_flax import types as types

__version__ = "0.8.5"


def enable_eager_mode(enabled: bool = True) -> None:
    """Configure the `SWITCHEROO_EAGER_MODE` environment flag.

    Args:
        enabled: If ``True``, sets ``SWITCHEROO_EAGER_MODE="1"`` for immediate
            numerical execution; if ``False``, sets ``SWITCHEROO_EAGER_MODE="0"``
            for symbolic graph tracing.
    """
    os.environ["SWITCHEROO_EAGER_MODE"] = "1" if enabled else "0"


__all__ = ["__version__", "enable_eager_mode", "nnx", "types"]
