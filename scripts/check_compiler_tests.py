"""Parity test coverage verifier for all exported `zero_flax.nnx` layers and transforms.

This script verifies that every core NNX layer, recurrent cell, normalization,
container, and transformation class exported in `zero_flax.nnx` has a dedicated,
non-vacuous parity test with value assertions in `tests/test_parity_flax.py`.
"""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Sequence

REQUIRED_NNX_LAYERS: tuple[str, ...] = (
    "BatchNorm",
    "Bidirectional",
    "Conv",
    "ConvTranspose",
    "Dense",
    "Dict",
    "Dropout",
    "Einsum",
    "Embed",
    "GRUCell",
    "GroupNorm",
    "InstanceNorm",
    "Jit",
    "LSTMCell",
    "LayerNorm",
    "Linear",
    "LinearGeneral",
    "List",
    "LoRA",
    "LoRALinear",
    "MultiHeadAttention",
    "MultiHeadDotProductAttention",
    "OptimizedLSTMCell",
    "PReLU",
    "Pmap",
    "RMSNorm",
    "RNN",
    "Remat",
    "RoPE",
    "Scan",
    "Sequential",
    "SimpleCell",
    "SpectralNorm",
    "Vmap",
    "WeightNorm",
)

VALUE_ASSERT_CALL_NAMES: frozenset[str] = frozenset(
    {
        "assert_allclose",
        "assert_array_equal",
        "assert_array_almost_equal",
        "assert_equal",
    }
)


def _is_shape_only_assert(assert_node: ast.Assert) -> bool:
    """Check whether an ``ast.Assert`` statement only inspects ``.shape``.

    Args:
        assert_node: AST ``Assert`` node to inspect.

    Returns:
        ``True`` if the assertion compares ``.shape`` without checking values.
    """
    test_expr = assert_node.test
    if isinstance(test_expr, ast.Compare) and isinstance(test_expr.left, ast.Attribute):
        return test_expr.left.attr == "shape"
    return False


def _has_value_assertion(fn_node: ast.FunctionDef) -> bool:
    """Check whether ``fn_node`` contains at least one non-shape value assertion or ``assert_allclose`` call.

    Args:
        fn_node: AST ``FunctionDef`` node of a test function.

    Returns:
        ``True`` if ``fn_node`` performs at least one value assertion, else ``False``.
    """
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Assert) and not _is_shape_only_assert(sub):
            return True
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name) and sub.func.id in VALUE_ASSERT_CALL_NAMES:
                return True
            if isinstance(sub.func, ast.Attribute) and sub.func.attr in VALUE_ASSERT_CALL_NAMES:
                return True
    return False


def check_parity_file(parity_test_path: str = "tests/test_parity_flax.py") -> list[str]:
    """Inspect ``parity_test_path`` to ensure every required NNX class has a non-empty test.

    Args:
        parity_test_path: File path to ``tests/test_parity_flax.py``.

    Returns:
        List of missing or vacuous test error messages.
    """
    if not os.path.exists(parity_test_path):
        return [f"Missing test file: {parity_test_path}"]

    with open(parity_test_path, encoding="utf-8") as handle:
        source = handle.read()

    tree = ast.parse(source, filename=parity_test_path)
    test_funcs: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_funcs[node.name.lower()] = node

    errors: list[str] = []
    for layer_name in REQUIRED_NNX_LAYERS:
        expected_fn = f"test_{layer_name.lower()}_parity"
        fn_node = test_funcs.get(expected_fn)
        if fn_node is None:
            errors.append(f"Missing parity test '{expected_fn}' for '{layer_name}'")
            continue
        non_doc_stmts = [
            stmt
            for stmt in fn_node.body
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        ]
        if (
            len(non_doc_stmts) == 0
            or all(isinstance(s, ast.Pass) for s in non_doc_stmts)
            or not _has_value_assertion(fn_node)
        ):
            errors.append(f"Vacuous 'pass' parity test in '{expected_fn}'")

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compiler/layer parity test verification check.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code ``0`` if all parity tests exist and are non-vacuous, ``1`` otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    target = args[0] if len(args) > 0 else "tests/test_parity_flax.py"
    errors = check_parity_file(target)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print(f"100% complete parity test suite verified ({len(REQUIRED_NNX_LAYERS)} NNX classes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
