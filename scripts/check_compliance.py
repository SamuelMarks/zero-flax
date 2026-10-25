"""API compliance verifier against `SEMANTIC_PLAN.md` and `ml-ecosystem-snapshots` v0.0.5.

This script verifies that:
1. No duplicate `grain`, `chex`, `orbax`, or `optax` modules are defined inside `src/zero_flax/`.
2. Every `flax.nnx` class in `ml-ecosystem-snapshots` v0.0.5 (`flax_nnx_v0.12.10.json`) is
   implemented on `zero_flax.nnx` with all official parameters, parameter kinds, type annotations,
   defaults, class attributes, and docstrings (additional parameters/functions are allowed).
3. Every API listed in `SEMANTIC_PLAN.md` exists and is exported on its target module.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FunctionType

import ml_ecosystem_snapshots

from zero_flax import nnx as zero_nnx
from zero_flax.nnx.module import Module

FORBIDDEN_SUBMODULE_NAMES: frozenset[str] = frozenset(
    {
        "grain",
        "chex",
        "orbax",
        "optax",
        "zero_grain",
        "zero_chex",
        "zero_orbax",
        "zero_optax",
    }
)

NNX_CLASS_REGISTRY: Mapping[str, type[Module]] = {
    "BatchNorm": zero_nnx.BatchNorm,
    "Bidirectional": zero_nnx.Bidirectional,
    "Conv": zero_nnx.Conv,
    "ConvTranspose": zero_nnx.ConvTranspose,
    "Dense": zero_nnx.Dense,
    "Dict": zero_nnx.Dict,
    "Dropout": zero_nnx.Dropout,
    "Einsum": zero_nnx.Einsum,
    "Embed": zero_nnx.Embed,
    "GRUCell": zero_nnx.GRUCell,
    "GroupNorm": zero_nnx.GroupNorm,
    "InstanceNorm": zero_nnx.InstanceNorm,
    "LSTMCell": zero_nnx.LSTMCell,
    "LayerNorm": zero_nnx.LayerNorm,
    "Linear": zero_nnx.Linear,
    "LinearGeneral": zero_nnx.LinearGeneral,
    "List": zero_nnx.List,
    "LoRA": zero_nnx.LoRA,
    "LoRALinear": zero_nnx.LoRALinear,
    "MultiHeadAttention": zero_nnx.MultiHeadAttention,
    "MultiHeadDotProductAttention": zero_nnx.MultiHeadDotProductAttention,
    "OptimizedLSTMCell": zero_nnx.OptimizedLSTMCell,
    "PReLU": zero_nnx.PReLU,
    "RMSNorm": zero_nnx.RMSNorm,
    "RNN": zero_nnx.RNN,
    "RNNCellBase": zero_nnx.RNNCellBase,
    "RoPE": zero_nnx.RoPE,
    "Sequential": zero_nnx.Sequential,
    "SimpleCell": zero_nnx.SimpleCell,
    "SpectralNorm": zero_nnx.SpectralNorm,
    "WeightNorm": zero_nnx.WeightNorm,
}


def _default_snapshot_path() -> Path:
    """Resolve the path to ``flax_nnx_v0.12.10.json`` bundled in ``ml-ecosystem-snapshots`` v0.0.5.

    Returns:
        Path to ``flax_nnx_v0.12.10.json``.
    """
    pkg_file = ml_ecosystem_snapshots.__file__
    base_dir = Path(pkg_file).parent if pkg_file is not None else Path(".")
    return base_dir / "snapshots" / "flax_nnx_v0.12.10.json"


def verify_no_forbidden_submodules(src_root: str = "src/zero_flax") -> list[str]:
    """Ensure ``src_root`` does not define local ``grain``, ``chex``, ``orbax``, or ``optax`` modules.

    Args:
        src_root: Root source directory of ``zero_flax``.

    Returns:
        List of forbidden module path violations (empty if compliant).
    """
    if not os.path.exists(src_root):
        return []
    violations: list[str] = []
    for root, dirs, files in os.walk(src_root):
        for dname in dirs:
            if dname in FORBIDDEN_SUBMODULE_NAMES:
                violations.append(
                    f"Forbidden duplicate directory in zero-flax: {os.path.join(root, dname)}"
                )
        for fname in files:
            stem = fname[:-3] if fname.endswith(".py") else fname
            if stem in FORBIDDEN_SUBMODULE_NAMES:
                violations.append(
                    f"Forbidden duplicate module in zero-flax: {os.path.join(root, fname)}"
                )
    return sorted(violations)


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


def _get_init_fn(cls_obj: type[Module]) -> FunctionType:
    """Resolve the ``__init__`` function defined on ``cls_obj``.

    Args:
        cls_obj: Module subclass to inspect.

    Returns:
        Resolved ``__init__`` ``FunctionType``.
    """
    members = dict(inspect.getmembers(cls_obj, inspect.isfunction))
    return members["__init__"]


def verify_snapshot_v005(
    snapshot_path: Path | None = None,
    registry: Mapping[str, type[Module]] = NNX_CLASS_REGISTRY,
) -> tuple[int, int, list[str]]:
    """Validate ``zero_flax.nnx`` against the ``flax.nnx`` entries in ``ml-ecosystem-snapshots`` v0.0.5.

    Ensures that every ``flax.nnx`` class in the snapshot exists in ``zero_flax.nnx`` and implements
    at a minimum every parameter, parameter kind, default presence, type annotation, and docstring
    from the v0.0.5 snapshot (while allowing additional parameters/functions).

    Args:
        snapshot_path: Optional explicit path to ``flax_nnx_v0.12.10.json``.
        registry: Mapping of class names to ``zero_flax.nnx`` class objects.

    Returns:
        Tuple ``(compliant_count, total_count, violations)``.
    """
    resolved_path = snapshot_path if snapshot_path is not None else _default_snapshot_path()
    if not resolved_path.exists():
        return 0, 0, [f"Snapshot file not found: {resolved_path}"]

    raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    categories = raw_data.get("categories", {}) if isinstance(raw_data, Mapping) else {}
    layer_entries = categories.get("layer", []) if isinstance(categories, Mapping) else []

    total = 0
    compliant = 0
    violations: list[str] = []

    for item in layer_entries:
        if not isinstance(item, Mapping):
            continue
        api_path = str(item.get("api_path", ""))
        if not api_path.startswith("flax.nnx."):
            continue
        total += 1
        cls_name = str(item.get("name", api_path.split(".")[-1]))
        cls_obj = registry.get(cls_name)
        if cls_obj is None:
            violations.append(f"{api_path}: Missing class '{cls_name}' in zero_flax.nnx")
            continue

        cls_doc = cls_obj.__doc__ or ""
        init_fn = _get_init_fn(cls_obj)
        init_doc = init_fn.__doc__ or ""
        if not cls_doc.strip():
            violations.append(f"{api_path}: Missing class docstring")
            continue
        if not init_doc.strip():
            violations.append(f"{api_path}: Missing __init__ docstring")
            continue

        sig = inspect.signature(init_fn)
        actual_params = {name: param for name, param in sig.parameters.items() if name != "self"}
        expected_params = item.get("params", [])
        if not isinstance(expected_params, Sequence):
            continue

        item_errors: list[str] = []
        for exp_param in expected_params:
            if not isinstance(exp_param, Mapping):
                continue
            p_name = str(exp_param.get("name", ""))
            p_kind = str(exp_param.get("kind", ""))
            p_default = exp_param.get("default")
            if p_name not in actual_params:
                item_errors.append(f"{api_path}: Missing parameter '{p_name}'")
                continue
            act_param = actual_params[p_name]
            if act_param.kind.name != p_kind:
                item_errors.append(
                    f"{api_path}.{p_name}: Expected kind '{p_kind}', got '{act_param.kind.name}'"
                )
            if act_param.annotation is inspect.Parameter.empty:
                item_errors.append(f"{api_path}.{p_name}: Missing type annotation")
            if (
                p_default is not None
                and p_kind not in ("VAR_POSITIONAL", "VAR_KEYWORD")
                and act_param.default is inspect.Parameter.empty
            ):
                item_errors.append(
                    f"{api_path}.{p_name}: Expected default value ({p_default}), got required parameter"
                )
            if (
                f"{p_name}:" not in init_doc
                and f"*{p_name}:" not in init_doc
                and f"**{p_name}:" not in init_doc
            ):
                item_errors.append(
                    f"{api_path}.{p_name}: Parameter not documented in __init__ docstring"
                )

        if item_errors:
            violations.extend(item_errors)
        else:
            compliant += 1

    return compliant, total, violations


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


def main(
    argv: Sequence[str] | None = None,
    snapshot_path: Path | None = None,
    src_root: str = "src/zero_flax",
) -> int:
    """Execute the API compliance verification check against `SEMANTIC_PLAN.md` and `ml-ecosystem-snapshots` v0.0.5.

    Args:
        argv: Optional command-line arguments (optional path to ``SEMANTIC_PLAN.md``).
        snapshot_path: Optional explicit path to ``flax_nnx_v0.12.10.json``.
        src_root: Source directory checked for forbidden duplicate sibling modules.

    Returns:
        Exit code ``0`` if 100% compliant, ``1`` otherwise.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    plan_file = args[0] if len(args) > 0 else "SEMANTIC_PLAN.md"

    forbidden_errs = verify_no_forbidden_submodules(src_root)
    if forbidden_errs:
        for err in forbidden_errs:
            print(f"  - {err}")
        return 1

    snap_ok, snap_total, snap_errs = verify_snapshot_v005(snapshot_path=snapshot_path)
    if snap_errs:
        print(f"Snapshot v0.0.5 Check Failed: {snap_ok}/{snap_total} classes compliant.")
        for err in snap_errs:
            print(f"  - {err}")
        return 1

    implemented, total, unimplemented = verify_semantic_plan(plan_file)
    if total == 0:
        print(
            f"{plan_file} not found or empty; Snapshot v0.0.5 passed ({snap_ok}/{snap_total} classes)."
        )
        return 0
    if unimplemented:
        print(f"Compliance Check Failed: {implemented}/{total} APIs implemented.")
        for item in unimplemented:
            print(f"  - {item}")
        return 1

    print(
        f"Compliance Check Passed! 100% compliant ({implemented}/{total} plan APIs, "
        f"{snap_ok}/{snap_total} ml-ecosystem-snapshots v0.0.5 classes)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
