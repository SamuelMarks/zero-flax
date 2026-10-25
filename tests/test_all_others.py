"""Tests for variable representations, module state updates, containers, and layer forward passes."""

from __future__ import annotations

import numpy as np
from zero_jax.nn import initializers

from zero_flax.nnx import (
    BatchNorm,
    Conv,
    Dense,
    Dict,
    Dropout,
    Embed,
    LayerNorm,
    List,
    MultiHeadAttention,
    RMSNorm,
    Sequential,
    split,
)
from zero_flax.nnx.module import Module, merge
from zero_flax.nnx.state import BatchStat, Param, Rng, State, Variable


def test_state_and_vars() -> None:
    """Verify Variable, Param, BatchStat, Rng repr, equality, replace(None), and State merge behavior."""
    v = Variable(1, a=2)
    assert v.value == 1
    assert v.has_metadata("a")
    assert not v.has_metadata("missing")
    assert v.get_metadata("a") == 2
    assert v.get_metadata("missing", 99) == 99
    assert v.metadata["a"] == 2
    assert repr(v) == "Variable(1)"
    assert v.same_value_and_metadata(Variable(1, a=2))
    assert not v.same_value_and_metadata(Param(1, a=2))
    assert not v.same_value_and_metadata(Variable(1, a=99))
    assert not v.same_value_and_metadata(Variable(True, a=2))

    v_opt: Variable[int | None] = Variable(5)
    v_replaced_none = v_opt.replace(None)
    assert v_replaced_none.value is None
    assert v_replaced_none.same_value_and_metadata(Variable(None))
    assert not v_replaced_none.same_value_and_metadata(v_opt)

    v_arr1 = Variable(np.array([1.0, 2.0], dtype=np.float32))
    v_arr2 = Variable(np.array([1.0, 2.0], dtype=np.float32))
    v_arr_diff_shape = Variable(np.array([[1.0, 2.0]], dtype=np.float32))
    assert v_arr1.same_value_and_metadata(v_arr2)
    assert not v_arr1.same_value_and_metadata(v_arr_diff_shape)

    p = Param(2)
    assert repr(p) == "Param(2)"

    b = BatchStat(3)
    assert repr(b) == "BatchStat(3)"

    r = Rng(4)
    assert repr(r) == "Rng(4)"

    s1 = State({"a": p})
    s2 = State({"b": b})
    s3 = merge(s1, s2)
    assert "a" in s3 and "b" in s3

    s_p, s_b, s_rest = s3.split(Param, BatchStat)
    assert "a" in s_p
    assert "b" in s_b
    assert len(s_rest) == 0


def test_module_and_graph() -> None:
    """Verify nested Module state extraction, updates, split, and merge."""

    class SubMod(Module):
        """Inner test module holding a single Param."""

        p: Param[int]

        def __init__(self) -> None:
            """Initialize SubMod with ``p = Param(1)``."""
            super().__init__()
            self.p = self.register_variable("p", Param(1))
            self.finish_init()

    class MainMod(Module):
        """Outer test module holding a child SubMod, a BatchStat, and a static int."""

        sub: SubMod
        b: BatchStat[int]

        def __init__(self) -> None:
            """Initialize MainMod."""
            super().__init__()
            self.sub = self.register_child("sub", SubMod())
            self.b = self.register_variable("b", BatchStat(2))
            self.register_static("c", 3)
            assert self.is_initializing()
            self.finish_init()
            assert not self.is_initializing()

    m = MainMod()
    st = m.state()
    assert isinstance(st["sub"], State)
    assert isinstance(st["sub"]["p"], Param)
    assert st["sub"]["p"].value == 1
    assert isinstance(st["b"], BatchStat)
    assert st["b"].value == 2

    m.update(State({"b": BatchStat(20), "sub": State({"p": Param(10)})}))
    assert m.b.value == 20
    assert m.sub.p.value == 10

    m.update([State({"b": 200}), State({"c": 300})])
    assert m.b.value == 200
    assert m._static_config["c"] == 300

    m.update(b=2000)
    assert m.b.value == 2000

    gdef, st_full = split(m)
    m2 = gdef.merge(st_full)
    assert m2.b.value == 2000
    assert m2.sub.p.value == 10
    assert m2._static_config["c"] == 300

    gdef2, st_p, st_b, st_rest = split(m, Param, BatchStat)
    m3 = gdef2.merge((st_p, st_b, st_rest))
    assert m3.b.value == 2000

    m4 = merge(gdef2, st_p, st_b, st_rest)
    assert m4.b.value == 2000


def test_containers() -> None:
    """Verify Dict, List, and Sequential container execution and mutation methods."""
    d: Dict[Param[int] | int] = Dict({"a": 1}, b=2)
    assert d["a"] == 1 and d["b"] == 2
    assert d.get("a") == 1
    assert d.get("missing", 42) == 42
    assert d.keys() == ["a", "b"]
    assert d.values() == [1, 2]
    assert d.items() == [("a", 1), ("b", 2)]
    assert d.pop("a") == 1
    assert d.pop("missing", 99) == 99
    del d["b"]
    assert len(d) == 0
    d["p"] = Param(10)
    d.update(State({"p": Param(25), "extra_stat": 7}))
    p_item = d["p"]
    assert isinstance(p_item, Param) and p_item.value == 25
    assert d._static_config["extra_stat"] == 7
    d.clear()
    assert len(d) == 0

    d2 = Dict([("a", 1)], b=2)
    assert d2["a"] == 1 and d2["b"] == 2
    d3 = Dict(b=2)
    assert d3["b"] == 2

    lst: List[Param[int] | int] = List([Param(1), 2])
    assert isinstance(lst[0], Param) and lst[0].value == 1
    lst.insert(1, Param(5))
    assert len(lst) == 3
    lst.set_at(2, Param(9))
    lst.update(State({"0": Param(100)}))
    first_item = lst[0]
    assert isinstance(first_item, Param) and first_item.value == 100
    popped = lst.pop()
    assert isinstance(popped, Param) and popped.value == 9
    lst.delete_at(0)
    assert len(lst) == 1
    lst.clear()
    assert len(lst) == 0

    lst2: List[int] = List()
    assert len(lst2) == 0

    def add_one(x: int | float) -> int | float:
        """Add 1 to ``x``.

        Args:
            x: Input number.

        Returns:
            Incremented number.
        """
        return x + 1

    def mul_two(x: int | float) -> int | float:
        """Multiply ``x`` by 2.

        Args:
            x: Input number.

        Returns:
            Doubled number.
        """
        return x * 2

    seq = Sequential(add_one, mul_two)
    assert seq(1) == 4


def test_layers_and_stochastic() -> None:
    """Verify numerical forward passes of Dense, Dropout, BatchNorm, LayerNorm, RMSNorm, Conv, Embed, and MHA."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    d = Dense(2, 3, kernel_init=initializers.ones, bias_init=initializers.ones)
    y = np.asarray(d(x))
    np.testing.assert_allclose(y, np.array([[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]], dtype=np.float32))

    d_no_bias = Dense(2, 3, use_bias=False, kernel_init=initializers.ones)
    y_nb = np.asarray(d_no_bias(x))
    np.testing.assert_allclose(y_nb, np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]], dtype=np.float32))

    drop = Dropout(0.5)
    np.testing.assert_allclose(np.asarray(drop(x, deterministic=True)), x)
    drop_stoch = np.asarray(drop(x, deterministic=False))
    assert np.all((drop_stoch == 0.0) | np.isclose(drop_stoch, x * 2.0))
    drop0 = Dropout(0.0)
    np.testing.assert_allclose(np.asarray(drop0(x, deterministic=False)), x)
    drop1 = Dropout(1.0)
    assert np.all(np.asarray(drop1(x, deterministic=False)) == 0.0)

    bn = BatchNorm(2, epsilon=0.0)
    bn_out = np.asarray(bn(x, use_running_average=False))
    np.testing.assert_allclose(
        bn_out, np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32), atol=1e-5
    )
    bn_eval = np.asarray(bn(x, use_running_average=True))
    assert bn_eval.shape == (2, 2)
    bn_no = BatchNorm(2, use_scale=False, use_bias=False, subtract_mean=False, epsilon=0.0)
    bn_no_out = np.asarray(bn_no(x, use_running_average=False))
    expected_rms_col = x / np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    np.testing.assert_allclose(bn_no_out, expected_rms_col, rtol=1e-5)

    ln = LayerNorm(2, epsilon=0.0)
    np.testing.assert_allclose(
        np.asarray(ln(x)),
        np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(ln(np.array([1.0, 3.0], dtype=np.float32))),
        np.array([-1.0, 1.0], dtype=np.float32),
        atol=1e-5,
    )

    rn = RMSNorm(2, epsilon=0.0)
    rn_1d = np.asarray(rn(np.array([3.0, 3.0], dtype=np.float32)))
    np.testing.assert_allclose(rn_1d, np.array([1.0, 1.0], dtype=np.float32), atol=1e-5)
    assert rn(x).shape == (2, 2)

    c = Conv(2, 2, (1, 1), kernel_init=initializers.ones, bias_init=initializers.zeros)
    c_out = np.asarray(c(np.ones((1, 2, 2, 2), dtype=np.float32)))
    np.testing.assert_allclose(c_out, np.full((1, 2, 2, 2), 2.0, dtype=np.float32))

    e = Embed(10, 2, embedding_init=initializers.ones)
    np.testing.assert_allclose(
        np.asarray(e(np.array([0, 1], dtype=np.int32))),
        np.ones((2, 2), dtype=np.float32),
    )

    mha = MultiHeadAttention(2, 2, kernel_init=initializers.ones, bias_init=initializers.zeros)
    np.testing.assert_allclose(
        np.asarray(mha(np.ones((2, 2), dtype=np.float32))),
        np.full((2, 2), 4.0, dtype=np.float32),
    )
