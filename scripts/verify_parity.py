"""Parity verification entrypoint script for the zero-flax repository.

This script verifies that `tests/test_parity_flax.py` exists and passes all
AST-level layer parity and value-assertion checks across all 35 NNX classes.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.check_compiler_tests import check_parity_file


def verify_parity(parity_test_path: str = "tests/test_parity_flax.py") -> bool:
    """Check that ``parity_test_path`` exists and contains non-vacuous value-asserting parity tests.

    Args:
        parity_test_path: Path to ``tests/test_parity_flax.py``.

    Returns:
        ``True`` if the parity test file exists and passes all 35 layer checks, ``False`` otherwise.
    """
    print("Verifying parity with flax...")
    if not os.path.exists(parity_test_path):
        return False
    errors = check_parity_file(parity_test_path)
    if errors:
        return False
    print("Parity verification passed.")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the parity verification check.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code ``0`` on success, ``1`` on failure.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    target = args[0] if len(args) > 0 else "tests/test_parity_flax.py"
    if not verify_parity(target):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
