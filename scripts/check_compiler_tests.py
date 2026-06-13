import ast
import os
import sys
import re


def main():
    # We want to ensure that every operation in ml_switcheroo_compiler.flax has a test in test_parity_flax.py
    compiler_flax = "../ml-switcheroo-compiler/src/ml_switcheroo_compiler/flax.py"
    if not os.path.exists(compiler_flax):
        print("Compiler flax.py not found, skipping.")
        sys.exit(0)

    with open(compiler_flax, "r", encoding="utf-8") as f:
        code = f.read()

    classes = re.findall(r"class\s+([A-Za-z0-9_]+)\s*\(", code)

    # Filter out base classes or non-operations
    skip = {
        "Module",
        "GraphDef",
        "State",
        "Variable",
        "Param",
        "BatchStat",
        "Rng",
        "Sequential",
        "List",
        "Dict",
        "Jit",
        "Vmap",
        "Scan",
        "Remat",
        "Pmap",
    }
    ops = [c for c in classes if c not in skip]

    test_file = "tests/test_parity_flax.py"
    if not os.path.exists(test_file):
        print(f"ERROR: {test_file} not found.")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        test_code = f.read()

    missing = []
    for op in ops:
        if (
            f"test_{op.lower()}_" not in test_code.lower()
            and f"_{op.lower()}" not in test_code.lower()
            and f"{op}" not in test_code
        ):
            missing.append(op)

    if missing:
        print(f"ERROR: Missing parity tests for operations: {missing}")
        sys.exit(1)

    print("100% complete test suite for compiler ops verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
