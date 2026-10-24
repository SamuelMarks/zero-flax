"""Pytest configuration and path setup for the zero-flax test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

import zero_flax

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC_PATH):
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)


@pytest.fixture(autouse=True)
def ensure_eager_execution() -> Iterator[None]:
    """Ensure ``SWITCHEROO_EAGER_MODE="1"`` is active before and after each test case.

    Yields:
        None during test execution.
    """
    zero_flax.enable_eager_mode(True)
    yield
    zero_flax.enable_eager_mode(True)
