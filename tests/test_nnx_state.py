"""Tests for zero_flax.nnx State class initialization and mutation tracking."""

from zero_flax.nnx import Module, Param, BatchStat, State, merge


class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.p1 = Param(1.0)
        self.b1 = BatchStat(0.0)
        self._is_initializing = False


def test_nnx_state_initialization():
    m = MyModule()
    state = m.state()
    assert "p1" in state
    assert "b1" in state
    assert state["p1"].value == 1.0


def test_nnx_mutation_tracking():
    m = MyModule()
    m.p1.value = 2.0
    state = m.state()
    assert state["p1"].value == 2.0

    # Test split and merge
    params, stats, rest = state.split(Param, BatchStat)
    assert "p1" in params
    assert "b1" in stats
    assert "p1" not in stats

    merged = merge(params, stats)
    assert "p1" in merged
    assert "b1" in merged


def test_nnx_state_coverage():
    from zero_flax.nnx.state import Variable, Param, State

    v = Variable(5)
    assert repr(v) == "Variable(5)"

    # State split remainder coverage
    s = State({"v": Variable(1), "n": 5})
    res = s.split(Param)
    assert len(res) == 2
    assert "n" in res[1]


def test_nnx_module_coverage():
    from zero_flax.nnx import Module, Param
    import pytest

    class M(Module):
        def __init__(self):
            super().__init__()
            self._x = 5
            self.p = Param(1)
            self.m2 = Module()
            self.m2.p2 = Param(2)
            super().__setattr__("_is_initializing", False)

    m = M()

    with pytest.raises(ValueError):
        m.new_attr = 5

    s = m.state()
    assert "p" in s
    assert "m2" in s
    assert "p2" in s["m2"]

    new_s = State({"p": Param(10), "m2": State({"p2": Param(20)})})
    m.update(new_s)
    assert m.p.value == 10
    assert m.m2.p2.value == 20
