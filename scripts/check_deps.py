"""Dependency isolation verifier for the zero-flax repository.

This script inspects Python files in `src/zero_flax` to ensure that only
Python standard library modules and approved `ml-switcheroo` / `zero-*`
ecosystem packages are imported at runtime.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import sysconfig
from collections.abc import Sequence

ALLOWED_3RD_PARTY: frozenset[str] = frozenset(
    {
        "pydantic",
        "cdd",
        "typing_extensions",
        "ml_switcheroo_ir",
        "ml_switcheroo",
        "ml_switcheroo_compiler",
        "ml_ecosystem_snapshots",
        "zero_jax",
        "zero_grain",
        "grain",
        "zero_orbax",
        "orbax",
        "zero_chex",
        "chex",
        "zero_optax",
        "optax",
        "zero_flax",
    }
)


def is_stdlib(module_name: str) -> bool:
    """Check whether ``module_name`` belongs to the Python standard library.

    Args:
        module_name: Import module name to check.

    Returns:
        ``True`` if ``module_name`` is a standard library or builtin module.
    """
    base_module = module_name.split(".")[0]
    if base_module in sys.builtin_module_names:
        return True

    spec = importlib.util.find_spec(base_module)
    if spec is None or spec.origin is None:
        return False
    stdlib_path = sysconfig.get_path("stdlib") or ""
    return spec.origin.startswith(stdlib_path) and "site-packages" not in spec.origin


def collect_src_files(paths: Sequence[str]) -> list[str]:
    """Collect ``.py`` files from ``paths`` (defaulting to ``src/zero_flax``).

    Args:
        paths: Sequence of file or directory paths.

    Returns:
        Sorted list of Python source file paths.
    """
    targets = list(paths) if len(paths) > 0 else ["src/zero_flax"]
    result: list[str] = []
    for path in targets:
        if os.path.isfile(path):
            result.append(path)
        else:
            for root, _dirs, files in os.walk(path):
                for fname in files:
                    if fname.endswith(".py"):
                        result.append(os.path.join(root, fname))
    return sorted(set(result))


def check_file(filepath: str) -> bool:
    """Verify that ``filepath`` contains no unauthorized third-party imports.

    Args:
        filepath: Path to the Python source file to inspect.

    Returns:
        ``True`` if all imports are allowed, ``False`` otherwise.
    """
    with open(filepath, encoding="utf-8") as handle:
        content = handle.read()

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as exc:
        print(f"SyntaxError in {filepath}: {exc}")
        return False

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".")[0])

    unauthorized: list[str] = []
    for mod in sorted(imports):
        if mod not in ALLOWED_3RD_PARTY and not is_stdlib(mod):
            unauthorized.append(mod)

    if unauthorized:
        print(f"Error in {filepath}: Unauthorized 3rd-party imports found: {unauthorized}")
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run dependency isolation checks across all target source files.

    Args:
        argv: Optional command-line file/directory arguments.

    Returns:
        Exit code ``0`` if all files pass, ``1`` otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    files = collect_src_files(args)
    all_passed = True
    for filepath in files:
        if not check_file(filepath):
            all_passed = False
    if not all_passed:
        return 1
    print(f"Dependency check passed across {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
