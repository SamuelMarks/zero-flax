"""Tests for variable representations, module state updates, containers, and layer forward passes."""

from __future__ import annotations

import numpy as np

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
    """Verify Variable, Param, BatchStat, Rng repr and State merge behavior."""
    v = Variable(1, a=2)
    assert v.value == 1
    assert v.has_metadata("a")
    assert not v.has_metadata("missing")
    assert v.get_metadata("a") == 2
    assert v.get_metadata("missing", 99) == 99
    assert v.metadata["a"] == 2
    assert repr(v) == "Variable(1)"

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
            self.finish_init()

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
    """Verify Dict, List, and Sequential container execution."""
    d = Dict({"a": 1}, b=2)
    assert d["a"] == 1 and d["b"] == 2
    d2 = Dict([("a", 1)], b=2)
    assert d2["a"] == 1 and d2["b"] == 2
    d3 = Dict(b=2)
    assert d3["b"] == 2

    lst = List([1, 2])
    assert lst[0] == 1
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
    """Verify forward passes of Dense, Dropout, BatchNorm, LayerNorm, RMSNorm, Conv, Embed, and MHA."""
    x = np.ones((2, 2), dtype=np.float32)

    d = Dense(2, 3)
    y = d(x)
    assert y.shape == (2, 3)

    d_no_bias = Dense(2, 3, use_bias=False)
    y_nb = d_no_bias(x)
    assert y_nb.shape == (2, 3)

    drop = Dropout(0.5)
    assert drop(x, deterministic=True).shape == (2, 2)
    assert drop(x, deterministic=False).shape == (2, 2)
    drop0 = Dropout(0.0)
    assert drop0(x, deterministic=False).shape == (2, 2)
    drop1 = Dropout(1.0)
    assert np.all(np.asarray(drop1(x, deterministic=False)) == 0.0)

    bn = BatchNorm(2)
    assert bn(x, use_running_average=False).shape == (2, 2)
    assert bn(x, use_running_average=True).shape == (2, 2)
    bn_no = BatchNorm(2, use_scale=False, use_bias=False, subtract_mean=False)
    assert bn_no(x, use_running_average=False).shape == (2, 2)

    ln = LayerNorm(2)
    assert ln(x).shape == (2, 2)
    assert ln(np.array([1.0, 2.0], dtype=np.float32)).shape == (2,)

    rn = RMSNorm(2)
    assert rn(x).shape == (2, 2)
    assert rn(np.array([1.0, 2.0], dtype=np.float32)).shape == (2,)

    c = Conv(2, 2, (1, 1))
    assert c(np.ones((1, 2, 2, 2), dtype=np.float32)).shape == (1, 2, 2, 2)

    e = Embed(10, 2)
    assert e(np.array([0, 1], dtype=np.int32)).shape == (2, 2)

    mha = MultiHeadAttention(2, 2)
    assert mha(x, x, x).shape == (2, 2)
