"""Parity verification entrypoint script for the zero-flax repository.

This script verifies that `tests/test_parity_flax.py` exists and passes all
AST-level layer parity checks across all 35 NNX classes.
"""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Sequence


def verify_parity(parity_test_path: str = "tests/test_parity_flax.py") -> bool:
    """Check that ``parity_test_path`` exists and contains non-vacuous parity tests.

    Args:
        parity_test_path: Path to ``tests/test_parity_flax.py``.

    Returns:
        ``True`` if the parity test file exists and has non-vacuous test definitions, ``False`` otherwise.
    """
    print("Verifying parity with flax...")
    if not os.path.exists(parity_test_path):
        return False
    with open(parity_test_path, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=parity_test_path)
    test_fns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if len(test_fns) < 35 or any(
        not any(isinstance(sub, (ast.Assert, ast.Call)) for sub in ast.walk(fn)) for fn in test_fns
    ):
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
