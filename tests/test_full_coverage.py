import pytest
import numpy as np
from jax import numpy as jnp
import zero_flax.nnx as nnx
from zero_flax.nnx.missing import filterlib, variables, rnglib
from zero_flax.nnx import state
from zero_flax.nnx.linear import Linear
from zero_flax.nnx.module import Module, GraphDef


def test_missing_classes():
    assert nnx.Dtype() is not None
    assert nnx.Shape() is not None
    assert nnx.Axes() is not None
    assert nnx.Size() is not None
    assert nnx.Axis() is not None
    assert nnx.DotGeneralT() is not None
    assert nnx.MaxFun() is not None
    assert filterlib.Filter() is not None
    assert rnglib.Rngs() is not None
    assert variables.Variable() is not None
    assert nnx.missing.AxisName() is not None
    assert nnx.missing.PrecisionLike() is not None
    assert nnx.missing.PaddingLike() is not None
    assert nnx.missing.Initializer() is not None


def test_missing_calls():
    assert nnx.Jit(lambda: None)() is None
    assert nnx.Remat(lambda: None)() is None
    assert nnx.Vmap(lambda: None)() is None
    assert nnx.Scan(lambda: None)() is None
    assert nnx.Pmap(lambda: None)() is None
    assert nnx.ConvTranspose(1, 1, (1, 1))() is None

    do = nnx.Dropout(0.5)
    np.testing.assert_array_equal(do(jnp.ones(1)), jnp.ones(1))

    from zero_flax.nnx.missing import (
        LoRA,
        LoRALinear,
        MultiHeadAttention,
        BatchNorm,
        Dropout as MissingDropout,
    )

    assert LoRA(1, 1, 1)() is None
    assert LoRALinear(1, 1, 1)() is None
    assert MultiHeadAttention(1, 1)() is None
    assert BatchNorm(1)() is None
    assert MissingDropout(0.5)() is None


def test_containers():
    d = nnx.Dict()
    d["k"] = 1
    assert d["k"] == 1
    d = nnx.Dict(k=2)
    assert d["k"] == 2
    assert d() is None

    l = nnx.List([1, 2])
    assert len(list(l)) == 2
    l[0] = 3
    assert l[0] == 3

    seq = nnx.Sequential(lambda x: x + 1, lambda x: x * 2)
    assert seq(1) == 4


def test_linear_no_bias():
    l = Linear(2, 2, use_bias=False)
    l.kernel = nnx.Param(jnp.ones((2, 2)))
    out = l(jnp.ones((1, 2)))
    assert out.shape == (1, 2)


def test_module_missing_attrs():
    class BadMod(nnx.Module):
        def __init__(self):
            super().__setattr__("_is_initializing", False)
            object.__setattr__(self, "x", 1)

    m = BadMod()
    m.x = 2
    assert m.x == 2

    s = m.state()
    assert len(s) == 0

    g, s2 = nnx.split(m)
    assert len(s2) == 0

    with pytest.raises(ValueError):
        m.new_attr = 1


def test_module_coverage():
    class M(nnx.Module):
        def __init__(self):
            super().__init__()
            self.x = 1
            self.y = nnx.Param(2)
            self.sub = nnx.Sequential()

    m = M()
    g, s = nnx.split(m)
    assert s is not None
    m2 = nnx.merge(g, s)
    assert m2.x == 1

    s = m.state()
    assert s["y"].value == 2

    s["y"].value = 3
    m.update(s)
    assert m.y.value == 3

    m.update(None)


def test_module_missing_paths():
    class M(nnx.Module):
        def __init__(self):
            super().__init__()
            self.p = nnx.Param(1)
            self.c = nnx.List()

    m = M()
    m.foo = "bar"
    assert m.foo == "bar"

    g, s = nnx.split(m)
    m2 = nnx.merge(g, s)
    assert m2.p.value == 1

    class N(nnx.Module):
        def __init__(self):
            super().__init__()
            self.x = 1

    n = N()
    n.p = nnx.Param(2)
    assert n.state()["p"].value == 2

    s2 = nnx.State({"unknown": nnx.Param(3)})
    n.update(s2)


def test_module_missing_paths2():
    class M(nnx.Module):
        def __init__(self):
            pass

    m = M()
    m.c = nnx.List()
    m.v = nnx.Param(1)
    assert hasattr(m, "_children")
    assert hasattr(m, "_variables")


def test_module_missing_paths3():
    class M(nnx.Module):
        def __init__(self):
            super().__init__()
            self.c = nnx.List()
            self.p = nnx.Param(1)

    m = M()
    g = nnx.GraphDef(M, {"c": nnx.GraphDef(nnx.List, {}, {})}, {})
    s = nnx.State({"c.somevar": nnx.Param(2)})

    m2 = g.merge(s)
    assert hasattr(m2, "c")

    g2, *s2 = nnx.split(m, nnx.Param)
    assert g2 is not None


def test_module_missing_paths4():
    class M(nnx.Module):
        def __init__(self):
            super().__init__()
            self.p = nnx.Param(1)

    m = M()
    g, s = nnx.split(m)
    m2 = g.merge((s,))
    assert m2.p.value == 1


def test_module_missing_paths5():
    class M(nnx.Module):
        def __init__(self):
            super().__init__()
            self.p = nnx.Param(1)

    m = M()
    g, s1, s2 = nnx.split(m, nnx.Param)


def test_merge_state_fallback():
    s1 = nnx.State({"a": nnx.Param(1)})
    s2 = nnx.State({"b": nnx.Param(2)})
    s_merged = nnx.merge(s1, s2)
    assert s_merged["a"].value == 1
    assert s_merged["b"].value == 2


def test_module_missing_paths6():
    m = nnx.Module()
    m.v1 = nnx.Param(1)
    m.v2 = nnx.BatchStat(2)
    g, s1, s2, s3 = nnx.split(m, nnx.Param, nnx.BatchStat)
    assert g is not None
    assert s1 is not None


def test_normalization():
    x = np.random.normal(size=(2, 3))

    ln = nnx.LayerNorm(3, rngs=rnglib.Rngs(0))
    assert ln(x).shape == (2, 3)

    rn = nnx.RMSNorm(3, rngs=rnglib.Rngs(0))
    assert rn(x).shape == (2, 3)
