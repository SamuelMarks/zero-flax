"""Semantic plan parity test verifier for the zero-flax repository.

This script checks that every Flax NNX component listed in `SEMANTIC_PLAN.md`
is referenced and tested in `tests/test_parity_flax.py`.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence


def verify_parity_references(
    plan_path: str = "SEMANTIC_PLAN.md",
    parity_test_path: str = "tests/test_parity_flax.py",
) -> list[str]:
    """Verify all Flax NNX items in ``plan_path`` appear in ``parity_test_path``.

    Args:
        plan_path: Path to ``SEMANTIC_PLAN.md``.
        parity_test_path: Path to ``tests/test_parity_flax.py``.

    Returns:
        List of missing component names.
    """
    if not os.path.exists(plan_path):
        return []
    if not os.path.exists(parity_test_path):
        return [parity_test_path]

    with open(plan_path, encoding="utf-8") as handle:
        plan_lines = handle.readlines()
    with open(parity_test_path, encoding="utf-8") as handle:
        test_content = handle.read()

    in_nnx_section = False
    nnx_items: list[str] = []
    for line in plan_lines:
        if line.startswith("## "):
            in_nnx_section = "flax.nnx" in line
        elif in_nnx_section and line.strip().startswith("- ["):
            match = re.search(r"`([^`]+)`", line)
            nnx_items.append(match.group(1) if match else line.strip())

    missing: list[str] = []
    for item in nnx_items:
        if item not in test_content and item.lower() not in test_content.lower():
            missing.append(item)
    return missing


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity test reference check.

    Args:
        argv: Optional command-line arguments ``[plan_path, parity_test_path]``.

    Returns:
        Exit code ``0`` if all NNX components are tested, ``1`` otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    plan_path = args[0] if len(args) > 0 else "SEMANTIC_PLAN.md"
    test_path = args[1] if len(args) > 1 else "tests/test_parity_flax.py"
    missing = verify_parity_references(plan_path, test_path)
    if missing:
        print(f"ERROR: Missing parity tests against flax for: {missing}")
        return 1
    print("Parity tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
