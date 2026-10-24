"""AST-based static typing and reflection linter for the zero-flax repository.

This script inspects Python files in `src/zero_flax`, `scripts`, and `tests` to
enforce:
- Zero imports or annotations of `typing.Any`
- Zero annotations or base classes using `object`
- Zero imports or calls of `typing.cast`
- Zero calls to `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`,
  `locals`, `eval`, or `exec` in `src/zero_flax`
- Zero dunder reflection attribute accesses (`__dict__`, `__setattr__`,
  `__getattribute__`, `__new__`, `__class__`) in `src/zero_flax`
- Zero `# type: ignore` comments
- Explicit parameter and return type annotations on every function and method
"""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Sequence

BANNED_REFLECTION_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "eval",
        "exec",
    }
)
BANNED_DUNDER_ATTRS: frozenset[str] = frozenset(
    {"__dict__", "__setattr__", "__getattribute__", "__new__", "__class__"}
)


def collect_python_files(paths: Sequence[str]) -> list[str]:
    """Collect all ``.py`` file paths from the provided files or directories.

    Args:
        paths: Sequence of directory or file paths to scan.

    Returns:
        Sorted list of discovered ``.py`` file paths.
    """
    collected: list[str] = []
    for path in paths:
        if os.path.isfile(path):
            collected.append(path)
        else:
            for root, _dirs, files in os.walk(path):
                for fname in files:
                    if fname.endswith(".py"):
                        collected.append(os.path.join(root, fname))
    return sorted(set(collected))


def check_file(filepath: str) -> list[str]:
    """Inspect a single Python file for typing, cast, and reflection violations.

    Args:
        filepath: Path to the Python source file to inspect.

    Returns:
        List of human-readable violation descriptions (empty if compliant).
    """
    violations: list[str] = []
    with open(filepath, encoding="utf-8") as handle:
        source = handle.read()

    norm_path = filepath.replace(chr(92), "/")
    is_src = norm_path.startswith("src/") or "/src/zero_flax/" in norm_path
    is_self_linter = norm_path.endswith("lint_typing.py")

    if not is_self_linter:
        for line_no, line in enumerate(source.splitlines(), start=1):
            if "# type:" + " ignore" in line:
                violations.append(f"{filepath}:{line_no}: forbidden type-ignore comment")

    tree = ast.parse(source, filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ("Any", "cast"):
                    violations.append(f"{filepath}:{node.lineno}: forbidden import '{alias.name}'")
        elif isinstance(node, ast.Name):
            if node.id == "Any":
                violations.append(f"{filepath}:{node.lineno}: forbidden usage of 'Any'")
            elif node.id == "object":
                violations.append(f"{filepath}:{node.lineno}: forbidden usage of 'object'")
        elif isinstance(node, ast.Attribute):
            if node.attr == "Any":
                violations.append(f"{filepath}:{node.lineno}: forbidden attribute '.Any'")
            if is_src and node.attr in BANNED_DUNDER_ATTRS:
                violations.append(
                    f"{filepath}:{node.lineno}: forbidden dunder access '.{node.attr}'"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "cast":
                    violations.append(f"{filepath}:{node.lineno}: forbidden call to 'cast()'")
                if is_src and node.func.id in BANNED_REFLECTION_CALLS:
                    violations.append(
                        f"{filepath}:{node.lineno}: forbidden reflection call '{node.func.id}()'"
                    )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "cast":
                violations.append(f"{filepath}:{node.lineno}: forbidden call to '.cast()'")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is None:
                violations.append(
                    f"{filepath}:{node.lineno}: function '{node.name}' missing return type annotation"
                )
            all_args = (
                list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            )
            for arg in all_args:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is None:
                    violations.append(
                        f"{filepath}:{node.lineno}: parameter '{arg.arg}' in '{node.name}' missing type annotation"
                    )
            if node.args.vararg is not None and node.args.vararg.annotation is None:
                violations.append(
                    f"{filepath}:{node.lineno}: *{node.args.vararg.arg} in '{node.name}' missing type annotation"
                )
            if node.args.kwarg is not None and node.args.kwarg.annotation is None:
                violations.append(
                    f"{filepath}:{node.lineno}: **{node.args.kwarg.arg} in '{node.name}' missing type annotation"
                )

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the typing and reflection linter across target paths.

    Args:
        argv: Optional sequence of command-line arguments (defaults to ``src/zero_flax``, ``scripts``, ``tests``).

    Returns:
        Exit code ``0`` if no violations are found, ``1`` otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    targets = args if len(args) > 0 else ["src/zero_flax", "scripts", "tests"]
    files = collect_python_files(targets)
    all_violations: list[str] = []
    for fpath in files:
        all_violations.extend(check_file(fpath))

    if all_violations:
        for v in all_violations:
            print(f"ERROR: {v}")
        return 1

    print(f"Typing lint passed across {len(files)} files (0 Any, 0 object, 0 cast, 0 reflection).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
