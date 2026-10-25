"""Tests for `zero_flax.nnx` `State` initialization, splitting, merging, and mutation tracking."""

from __future__ import annotations

from pathlib import Path

import pytest
import zero_chex
import zero_optax
import zero_orbax.checkpoint as ocp
from ml_switcheroo_compiler.core.tensor import Tensor
from zero_chex._src.tree_util import tree_leaves as chex_tree_leaves
from zero_jax import numpy as jnp
from zero_optax._tree import asarray as optax_asarray
from zero_optax._tree import tree_map as optax_tree_map
from zero_optax.base import ExtraArgValue

from zero_flax.core import FrozenDict, freeze
from zero_flax.nnx import (
    BatchStat,
    Linear,
    LoRAParam,
    Module,
    Optimizer,
    OptState,
    OptVariable,
    Param,
    RngCount,
    State,
    merge,
    restore_state,
    save_state,
)
from zero_flax.nnx.state import (
    _coerce_leaf_to_variable_value,
    _coerce_metadata_dict,
    _state_to_chex_pytree,
)


class MyModule(Module):
    """Sample module containing a `Param`, a `BatchStat`, and a static integer."""

    p: Param[int]
    b: BatchStat[int]
    static: int

    def __init__(self) -> None:
        """Initialize MyModule with ``p=1``, ``b=2``, and ``static=3``."""
        super().__init__()
        self.p = self.register_variable("p", Param(1))
        self.b = self.register_variable("b", BatchStat(2))
        self.static = 3
        self.finish_init()


def test_state_extraction() -> None:
    """Verify Module.state() extracts Variables while excluding static fields."""
    m = MyModule()
    s = m.state()
    assert "p" in s
    assert "b" in s
    assert "static" not in s
    assert isinstance(s["p"], Param)
    assert s["p"].value == 1
    assert isinstance(s["b"], BatchStat)
    assert s["b"].value == 2


def test_state_mutation() -> None:
    """Verify Module.update() mutates existing Variable values in-place."""
    m = MyModule()
    s = m.state()
    s["p"] = Param(10)
    m.update(s)
    assert m.p.value == 10


def test_state_split_and_merge() -> None:
    """Verify State.split() and merge() partition and recombine Variable types."""
    m = MyModule()
    s = m.state()
    params, batch_stats, rest = s.split(Param, BatchStat)
    assert "p" in params
    assert "b" not in params
    assert "b" in batch_stats
    assert "p" not in batch_stats
    assert len(rest) == 0

    merged = merge(params, batch_stats, rest)
    assert "p" in merged
    assert "b" in merged


def test_nested_module() -> None:
    """Verify hierarchical State extraction and update on nested Modules."""

    class ParentModule(Module):
        """Parent module wrapping a child MyModule."""

        p: Param[int]
        m2: MyModule

        def __init__(self) -> None:
            """Initialize ParentModule with ``p=0`` and child ``m2``."""
            super().__init__()
            self.p = self.register_variable("p", Param(0))
            self.m2 = self.register_child("m2", MyModule())
            self.finish_init()

    m = ParentModule()
    s = m.state()
    assert "p" in s
    assert "m2" in s
    assert isinstance(s["m2"], State)
    assert "p" in s["m2"]

    new_s = State({"p": Param(10), "m2": State({"p": Param(20)})})
    m.update(new_s)
    assert m.p.value == 10
    assert m.m2.p.value == 20


def test_state_strict_shapes_and_chex_integration() -> None:
    """Verify Module.update(strict_shapes=True) and zero-chex PyTree traversal on State and FrozenDict."""
    lin = Linear(2, 3)
    st = lin.state()
    leaves = chex_tree_leaves(st)
    assert len(leaves) == 2
    zero_chex.assert_trees_all_close(st, lin.state())

    fd = freeze({"w": jnp.ones((2, 3)), "b": jnp.zeros((3,))})
    assert isinstance(fd, FrozenDict)
    zero_chex.assert_trees_all_equal_shapes(
        fd, freeze({"w": jnp.zeros((2, 3)), "b": jnp.ones((3,))})
    )

    valid_update = lin.state()
    lin.update(valid_update, kernel=Param(jnp.ones((2, 3))), strict_shapes=True)
    lin.update(valid_update, strict_shapes=True)

    bad_state = State({"kernel": Param(jnp.ones((4, 5))), "bias": Param(jnp.zeros((3,)))})
    with pytest.raises(AssertionError):
        lin.update(bad_state, strict_shapes=True)


def test_state_optax_and_pure_dict_conversions() -> None:
    """Verify State.to_optax_tree, from_optax_tree, to_pure_dict, and from_pure_dict across all branches."""
    raw_tensor = optax_asarray([1.0, 2.0])
    assert isinstance(raw_tensor, Tensor)
    assert isinstance(_coerce_leaf_to_variable_value(raw_tensor), jnp.ndarray)
    assert _coerce_leaf_to_variable_value(Param(3.0)) == 3.0
    assert isinstance(_coerce_leaf_to_variable_value([1.0, 2.0]), jnp.ndarray)
    assert _coerce_leaf_to_variable_value(None) is None
    coerced_arr = _coerce_leaf_to_variable_value(jnp.array([1.0, 2.0]))
    assert isinstance(coerced_arr, jnp.ndarray) and coerced_arr.shape == (2,)
    assert _coerce_metadata_dict(None) == {}
    assert _coerce_metadata_dict({"a": 1, "b": [1, 2], "c": ["x", "y"], "d": {"skip": 1}}) == {
        "a": 1,
        "b": (1, 2),
        "c": ["x", "y"],
    }

    st = State(
        {
            "layer": State(
                {
                    "w": Param(jnp.ones((2, 2)), sharding=("data",)),
                    "b": BatchStat([1.0, 2.0]),
                    "none_v": Param(None),
                    "scalar_v": Param(2.5),
                    "str_v": Param("name"),
                    "cnt": RngCount(3),
                }
            ),
            "raw_none": None,
            "raw_scalar": 4,
            "raw_str": "hello",
            "raw_seq": [1.0, 2.0],
        }
    )
    chex_tree = _state_to_chex_pytree(st)
    assert isinstance(chex_tree["layer"], dict)

    optax_tree = st.to_optax_tree()
    assert isinstance(optax_tree["layer"], dict)

    restored_from_optax = State.from_optax_tree(optax_tree, template=st)
    layer_optax = restored_from_optax["layer"]
    assert isinstance(layer_optax, State)
    assert isinstance(layer_optax["w"], Param)
    assert isinstance(layer_optax["b"], BatchStat)

    restored_no_tmpl = State.from_optax_tree(
        State({"a": State({"x": Param(jnp.ones((2,)))}), "y": jnp.zeros((1,))})
    )
    assert isinstance(restored_no_tmpl["a"], State)
    assert isinstance(restored_no_tmpl["y"], Param)

    pure_raw = st.to_pure_dict(include_variable_metadata=False)
    assert isinstance(pure_raw["layer"], dict)
    restored_from_raw = State.from_pure_dict(pure_raw, template=st)
    layer_raw = restored_from_raw["layer"]
    assert isinstance(layer_raw, State)
    assert isinstance(layer_raw["w"], Param)

    pure_meta = st.to_pure_dict(include_variable_metadata=True)
    restored_self_desc = State.from_pure_dict(pure_meta, template=None)
    layer_self_desc = restored_self_desc["layer"]
    assert isinstance(layer_self_desc, State)
    assert isinstance(layer_self_desc["w"], Param)
    assert isinstance(layer_self_desc["b"], BatchStat)
    assert isinstance(layer_self_desc["cnt"], RngCount)

    restored_self_with_tmpl = State.from_pure_dict(pure_meta, template=st)
    layer_with_tmpl = restored_self_with_tmpl["layer"]
    assert isinstance(layer_with_tmpl, State)
    assert isinstance(layer_with_tmpl["w"], Param)

    restored_var_direct = State.from_pure_dict(
        {
            "v1": Param(jnp.ones((2,))),
            "v2": LoRAParam(jnp.zeros((2,))),
            "nested": {"__zero_flax_state__": True, "entries": {"k": jnp.ones((1,))}},
            "plain": jnp.ones((1,)),
            "sub": State({"z": Param(1.0)}),
            "str_cnt": {"__variable_type__": "RngCount", "value": "not_int", "metadata": {}},
        },
        template=State({"v1": Param(jnp.zeros((2,)))}),
    )
    assert isinstance(restored_var_direct["v1"], Param)
    assert isinstance(restored_var_direct["v2"], LoRAParam)
    assert isinstance(restored_var_direct["nested"], State)
    assert isinstance(restored_var_direct["plain"], Param)
    assert isinstance(restored_var_direct["sub"], State)


def test_nnx_optimizer_all_12_optax_optimizers_and_orbax_roundtrip(tmp_path: Path) -> None:
    """Verify nnx.Optimizer with all 12 zero-optax optimizers, schedules, clipping, and zero-orbax checkpointing.

    Args:
        tmp_path: Pytest temporary directory fixture for checkpoint storage.
    """
    optimizer_transforms: list[zero_optax.GradientTransformation[zero_optax.OptState]] = [
        zero_optax.sgd(learning_rate=0.05, momentum=0.9),
        zero_optax.adam(learning_rate=0.01),
        zero_optax.adamw(learning_rate=0.01, weight_decay=1e-4),
        zero_optax.rmsprop(learning_rate=0.01),
        zero_optax.adagrad(learning_rate=0.05),
        zero_optax.adabelief(learning_rate=0.01),
        zero_optax.lion(learning_rate=0.01),
        zero_optax.lamb(learning_rate=0.01),
        zero_optax.lars(learning_rate=0.01),
        zero_optax.fromage(learning_rate=0.01),
        zero_optax.novograd(learning_rate=0.01),
        zero_optax.yogi(learning_rate=0.01),
    ]

    for tx_item in optimizer_transforms:
        lin = Linear(2, 2)
        opt = Optimizer(lin, tx_item)
        assert isinstance(opt.step, OptState)
        assert isinstance(opt.opt_variable, OptVariable)
        grads = State(
            {
                "kernel": Param(jnp.full((2, 2), 0.1)),
                "bias": Param(jnp.full((2,), 0.05)),
            }
        )
        opt.update(grads)
        assert int(jnp.asarray(opt.step.value).item()) == 1

    sched = zero_optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.05,
        warmup_steps=2,
        decay_steps=10,
        end_value=0.001,
    )
    chained_tx = zero_optax.chain(
        zero_optax.clip_by_global_norm(1.0),
        zero_optax.adamw(learning_rate=sched),
    )
    model = Linear(2, 2)
    opt_chained = Optimizer(model, chained_tx)
    opt_chained.update(
        {
            "kernel": Param(jnp.full((2, 2), 0.2)),
            "bias": Param(jnp.full((2,), 0.1)),
        }
    )

    def _extra_init(params: zero_optax.Params) -> zero_optax.EmptyState:
        """Initialize EmptyState for extra_args test.

        Args:
            params: Parameter PyTree (unused).

        Returns:
            Initialized ``EmptyState``.
        """
        del params
        return zero_optax.EmptyState()

    def _extra_update(
        updates: zero_optax.Updates,
        state: zero_optax.OptState,
        params: zero_optax.Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[zero_optax.Updates, zero_optax.EmptyState]:
        """Scale updates by extra_args['scale'] for extra_args test.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.
            **extra_args: Extra keyword arguments containing ``scale``.

        Returns:
            Pair ``(scaled_updates, EmptyState())``.
        """
        del state, params
        scale_val = extra_args.get("scale", 1.0)
        scale_f = float(scale_val) if isinstance(scale_val, (int, float)) else 1.0
        scaled = optax_tree_map(lambda u: optax_asarray(u) * (-scale_f), updates)
        return scaled, zero_optax.EmptyState()

    extra_tx: zero_optax.GradientTransformationExtraArgs[zero_optax.EmptyState] = (
        zero_optax.GradientTransformationExtraArgs(_extra_init, _extra_update)
    )
    opt_extra = Optimizer(Linear(2, 2), extra_tx)
    opt_extra.update(
        State({"kernel": Param(jnp.ones((2, 2))), "bias": Param(jnp.ones((2,)))}),
        scale=0.1,
    )

    sd = opt_chained.state_dict()
    ckpt_dir = tmp_path / "opt_ckpt"
    pytree_ckptr = ocp.PyTreeCheckpointer()
    pytree_ckptr.save(ckpt_dir, sd)
    restored_sd = pytree_ckptr.restore(ckpt_dir)

    fresh_model = Linear(2, 2)
    fresh_opt = Optimizer(fresh_model, chained_tx)
    if isinstance(restored_sd, dict):
        fresh_opt.load_state_dict(restored_sd)
    fresh_opt.load_state_dict({"model_state": fresh_model.state(), "step": 2, "opt_state": None})
    fresh_opt.load_state_dict(
        {"model_state": fresh_model.state().to_pure_dict(), "step": 2, "opt_state": None}
    )
    fresh_opt.load_state_dict({"model_state": None, "step": "bad", "opt_state": None})
    zero_chex.assert_trees_all_close(fresh_model.state(), model.state())

    state_dir = tmp_path / "state_ckpt"
    assert save_state(state_dir, model.state())
    restored_no_target = restore_state(state_dir)
    target_state = Linear(2, 2).state()
    restored_with_target = restore_state(state_dir, target=target_state)
    zero_chex.assert_trees_all_close(
        restored_no_target.to_pure_dict(), model.state().to_pure_dict()
    )
    zero_chex.assert_trees_all_close(restored_with_target, model.state())
