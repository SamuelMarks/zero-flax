import sys
import os
import ast
import importlib.util
import sysconfig

ALLOWED_3RD_PARTY = {
    "numpy",
    "pydantic",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo",
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


def is_stdlib(module_name):
    if module_name in sys.builtin_module_names:
        return True

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        return False

    if spec is None:
        return False

    if spec.origin == "built-in":
        return True

    if spec.origin is None:
        return False

    stdlib_path = sysconfig.get_path("stdlib")
    if "site-packages" in spec.origin or "dist-packages" in spec.origin:
        return False

    return spec.origin.startswith(stdlib_path)


def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return True  # skip invalid files

    disallowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if not (top_level in ALLOWED_3RD_PARTY or is_stdlib(top_level)):
                    disallowed.add(top_level)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                top_level = node.module.split(".")[0]
                if not (top_level in ALLOWED_3RD_PARTY or is_stdlib(top_level)):
                    disallowed.add(top_level)

    if disallowed:
        print(
            f"{filepath}: Disallowed 3rd-party dependencies found: {', '.join(disallowed)}"
        )
        print(f"Allowed 3rd-party dependencies: {', '.join(sorted(ALLOWED_3RD_PARTY))}")
        return False
    return True


if __name__ == "__main__":
    all_passed = True
    for filepath in sys.argv[1:]:
        if not check_file(filepath):
            all_passed = False

    if not all_passed:
        sys.exit(1)
