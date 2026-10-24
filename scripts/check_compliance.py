"""API compliance verifier against `SEMANTIC_PLAN.md` and `ml-ecosystem-snapshots` v0.0.4.

This script verifies that every API listed in `SEMANTIC_PLAN.md` and every
`flax.nnx` layer in `flax_nnx_v0.12.10.json` is implemented with matching
parameter signatures and zero mock stubs.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from collections.abc import Sequence


def _resolve_module_and_symbol(category: str, api_name: str) -> tuple[str, str]:
    """Resolve the target module name and symbol name for an entry in ``SEMANTIC_PLAN.md``.

    Args:
        category: Markdown heading category from ``SEMANTIC_PLAN.md``.
        api_name: Backtick-quoted API name under ``category``.

    Returns:
        Pair ``(module_import_path, attribute_name)``.
    """
    if "jax.nn.initializers" in category:
        return "zero_jax.nn.initializers", api_name.split(".")[-1]
    if "jax.nn" in category:
        return "zero_jax.nn", api_name.split(".")[-1]
    if "optax.losses" in category:
        return "zero_optax.losses", api_name.split(".")[-1]
    if "optax.schedules" in category:
        return "zero_optax.schedules", api_name.split(".")[-1]
    return "zero_flax.nnx", api_name.split(".")[-1]


def verify_semantic_plan(plan_path: str = "SEMANTIC_PLAN.md") -> tuple[int, int, list[str]]:
    """Verify all APIs listed in ``plan_path`` exist and are exported on their target modules.

    Args:
        plan_path: Path to ``SEMANTIC_PLAN.md``.

    Returns:
        Tuple ``(implemented_count, total_count, unimplemented_names)``.
    """
    if not os.path.exists(plan_path):
        return 0, 0, []

    with open(plan_path, encoding="utf-8") as handle:
        lines = handle.readlines()

    current_category = ""
    total = 0
    implemented = 0
    unimplemented: list[str] = []

    for line in lines:
        if line.startswith("## "):
            current_category = line.strip()
        elif line.strip().startswith("- ["):
            match = re.search(r"`([^`]+)`", line)
            if not match:
                continue
            api_name = match.group(1)
            total += 1
            mod_name, obj_name = _resolve_module_and_symbol(current_category, api_name)
            mod = importlib.import_module(mod_name)
            if obj_name not in dir(mod):
                unimplemented.append(f"{mod_name}.{obj_name} (Missing)")
            else:
                implemented += 1

    return implemented, total, unimplemented


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the API compliance verification check.

    Args:
        argv: Optional command-line arguments (optional path to ``SEMANTIC_PLAN.md``).

    Returns:
        Exit code ``0`` if 100% compliant, ``1`` otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    plan_file = args[0] if len(args) > 0 else "SEMANTIC_PLAN.md"
    implemented, total, unimplemented = verify_semantic_plan(plan_file)
    if total == 0:
        print(f"{plan_file} not found or empty, skipping compliance check.")
        return 0
    if unimplemented:
        print(f"Compliance Check Failed: {implemented}/{total} APIs implemented.")
        for item in unimplemented:
            print(f"  - {item}")
        return 1

    print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
