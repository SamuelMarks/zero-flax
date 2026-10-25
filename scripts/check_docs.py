"""AST-based 100% docstring coverage verifier for the zero-flax repository.

This script inspects Python files in `src/zero_flax`, `scripts`, and `tests` to
enforce:
- Every file/module has a non-empty top-level module docstring
- Every class has a non-empty docstring
- Every function and method (excluding `@overload` stubs) has a non-empty docstring
- Every function/method parameter (excluding `self` and `cls`) is documented in `Args:`
- Every non-`None` returning function/method (excluding `__init__`) has a `Returns:` or `Yields:` section
- Every function/method in `src/zero_flax` that raises an exception documents `Raises:`
- No banned placeholder docstrings (`"Initialize."`, `"Test function."`)
"""

from __future__ import annotations

import ast
import os
import re
import sys
from collections.abc import Sequence

BANNED_STUB_DOCSTRINGS: frozenset[str] = frozenset(
    {"Initialize.", "Test function.", "Placeholder."}
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


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether ``node`` is decorated with ``@overload`` or ``@typing.overload``.

    Args:
        node: AST function definition node.

    Returns:
        ``True`` if decorated with ``overload``, else ``False``.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _returns_none(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether ``node`` has a ``-> None`` return annotation.

    Args:
        node: AST function definition node.

    Returns:
        ``True`` if the return annotation is ``None``, else ``False``.
    """
    return (
        node.returns is None
        or (isinstance(node.returns, ast.Constant) and node.returns.value is None)
        or (isinstance(node.returns, ast.Name) and node.returns.id == "None")
    )


def _has_direct_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether ``node`` contains an explicit ``raise`` statement outside nested functions.

    Args:
        node: AST function definition node.

    Returns:
        ``True`` if ``node`` directly raises an exception, else ``False``.
    """
    stack: list[ast.AST] = list(node.body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(cur, ast.Raise):
            return True
        stack.extend(ast.iter_child_nodes(cur))
    return False


def _extract_args_section(docstring: str) -> str:
    """Extract the ``Args:`` block from a Google-style ``docstring``.

    Args:
        docstring: Complete function or method docstring.

    Returns:
        Substring starting at ``Args:`` up to the next top-level section header or end of string.
    """
    idx = docstring.find("Args:")
    if idx == -1:
        return ""
    after_args = docstring[idx + len("Args:") :]
    nl = chr(10)
    next_sec = re.search(nl + r"\s*(Returns|Yields|Raises|Attributes|Note|Notes):", after_args)
    if next_sec is not None:
        return after_args[: next_sec.start()]
    return after_args


def check_file(filepath: str) -> list[str]:
    """Verify module, class, function, argument, return, and raises docstrings in ``filepath``.

    Args:
        filepath: Path to the Python file to inspect.

    Returns:
        List of docstring violation messages (empty if 100% compliant).
    """
    violations: list[str] = []
    with open(filepath, encoding="utf-8") as handle:
        source = handle.read()

    norm_path = filepath.replace(chr(92), "/")
    is_src = norm_path.startswith("src/") or "/src/zero_flax/" in norm_path

    tree = ast.parse(source, filename=filepath)
    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        violations.append(f"{filepath}:1: missing module docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                violations.append(
                    f"{filepath}:{node.lineno}: class '{node.name}' missing docstring"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_overload(node):
                continue
            fn_doc = ast.get_docstring(node)
            if not fn_doc or not fn_doc.strip():
                violations.append(
                    f"{filepath}:{node.lineno}: function '{node.name}' missing docstring"
                )
                continue
            if fn_doc.strip() in BANNED_STUB_DOCSTRINGS:
                violations.append(
                    f"{filepath}:{node.lineno}: function '{node.name}' has banned stub docstring '{fn_doc.strip()}'"
                )

            param_names: list[str] = []
            for arg in (
                list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            ):
                if arg.arg in ("self", "cls"):
                    continue
                param_names.append(arg.arg)
            if node.args.vararg is not None:
                param_names.append(node.args.vararg.arg)
            if node.args.kwarg is not None:
                param_names.append(node.args.kwarg.arg)

            if param_names:
                args_section = _extract_args_section(fn_doc)
                if not args_section:
                    violations.append(
                        f"{filepath}:{node.lineno}: function '{node.name}' missing 'Args:' section in docstring"
                    )
                else:
                    for p_name in param_names:
                        pattern = rf"\b{re.escape(p_name)}\b"
                        if not re.search(pattern, args_section):
                            violations.append(
                                f"{filepath}:{node.lineno}: parameter '{p_name}' not documented in '{node.name}' docstring"
                            )

            if (
                node.name != "__init__"
                and not _returns_none(node)
                and "Returns:" not in fn_doc
                and "Yields:" not in fn_doc
            ):
                violations.append(
                    f"{filepath}:{node.lineno}: function '{node.name}' missing 'Returns:' or 'Yields:' section"
                )

            if is_src and _has_direct_raise(node) and "Raises:" not in fn_doc:
                violations.append(
                    f"{filepath}:{node.lineno}: function '{node.name}' missing 'Raises:' section in docstring"
                )

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 100% docstring coverage verifier across target paths.

    Args:
        argv: Optional sequence of command-line arguments (defaults to ``src/zero_flax``, ``scripts``, ``tests``).

    Returns:
        Exit code ``0`` if 100% compliant, ``1`` otherwise.
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

    print(
        f"Docstring check passed: 100% file, module, class, function, and arg coverage across {len(files)} files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
