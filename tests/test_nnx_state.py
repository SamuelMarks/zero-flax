"""Tests for `zero_flax.nnx` `State` initialization, splitting, merging, and mutation tracking."""

from __future__ import annotations

from zero_flax.nnx import BatchStat, Module, Param, State, merge


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
