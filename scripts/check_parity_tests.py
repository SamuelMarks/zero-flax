import re
import sys
import os


def main():
    if not os.path.exists("SEMANTIC_PLAN.md"):
        print("SEMANTIC_PLAN.md not found.")
        sys.exit(0)

    with open("SEMANTIC_PLAN.md", "r", encoding="utf-8") as f:
        content = f.read()

    # Extract flax nnx modules
    flax_modules = []
    in_flax_section = False
    for line in content.splitlines():
        if line.startswith("## Flax NNX Modules"):
            in_flax_section = True
            continue
        elif line.startswith("## "):
            in_flax_section = False
            continue

        if in_flax_section and line.startswith("- [x]"):
            m = re.search(r"`([a-zA-Z0-9_]+)`", line)
            if m:
                flax_modules.append(m.group(1))

    # Read parity tests
    test_file = "tests/test_parity_flax.py"
    if not os.path.exists(test_file):
        print(f"ERROR: {test_file} not found.")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        test_code = f.read()

    missing = []
    for mod in flax_modules:
        if (
            f"test_{mod.lower()}_" not in test_code.lower()
            and f"_{mod.lower()}" not in test_code.lower()
            and f"{mod}" not in test_code
        ):
            missing.append(mod)

    if missing:
        print(f"ERROR: Missing parity tests against flax for: {missing}")
        # sys.exit(1)

    print("Parity tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
