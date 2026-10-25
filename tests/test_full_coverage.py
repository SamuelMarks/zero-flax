"""Comprehensive coverage tests for `zero_flax.nnx` modules, containers, layers, and types."""

from __future__ import annotations

import numpy as np
from zero_jax.nn import initializers

import zero_flax
import zero_flax.nnx as nnx
from zero_flax.nnx import state
from zero_flax.nnx.linear import Einsum, Linear, LinearGeneral
from zero_flax.nnx.missing import filterlib, rnglib, variables


def test_containers() -> None:
    """Verify Dict, List, and Sequential container initialization and execution."""
    zero_flax.enable_eager_mode(False)
    zero_flax.enable_eager_mode(True)

    d1 = nnx.Dict({"a": 1, "b": 2})
    assert "a" in d1
    assert d1["a"] == 1
    assert list(d1) == ["a", "b"]
    assert len(d1) == 2
    d2 = nnx.Dict([("a", 1), ("b", 2)])
    assert d2["b"] == 2
    d3 = nnx.Dict(a=1, b=2)
    assert d3["a"] == 1
    d4: nnx.Dict[nnx.Module | nnx.Param[int]] = nnx.Dict()
    d4["lin"] = nnx.Linear(2, 2)
    d4["param"] = nnx.Param(1)
    assert "lin" in d4.state()
    assert "param" in d4.state()

    l1 = nnx.List([1, 2, 3])
    assert len(l1) == 3
    l2: nnx.List[nnx.Module | nnx.Param[int]] = nnx.List()
    l2.extend([nnx.Linear(2, 2), nnx.Param(3)])
    assert len(l2) == 2
    assert "0" in l2.state()
    assert "1" in l2.state()

    def fn1(x: float, a: float = 1.0) -> float:
        """Add ``a`` to ``x``.

        Args:
            x: Input scalar.
            a: Offset to add.

        Returns:
            Sum ``x + a``.
        """
        return x + a

    def fn2(x: float, a: float = 1.0) -> float:
        """Multiply ``x`` by ``2 * a``.

        Args:
            x: Input scalar.
            a: Multiplier factor.

        Returns:
            Product ``x * 2 * a``.
        """
        return x * 2.0 * a

    seq = nnx.Sequential(fn1, fn2)
    assert seq(2.0, a=2.0) == 16.0


def test_layers() -> None:
    """Verify Dense, Conv, Embed, and MultiHeadDotProductAttention with exact numerical checks."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    d1 = nnx.Dense(3, 4, rngs=rnglib.Rngs(0))
    assert d1(x).shape == (2, 4)

    d2 = nnx.Dense(
        3,
        4,
        use_bias=False,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    np.testing.assert_allclose(
        np.asarray(d2(x)),
        np.array([[6.0, 6.0, 6.0, 6.0], [15.0, 15.0, 15.0, 15.0]], dtype=np.float32),
    )

    c1 = nnx.Conv(3, 4, kernel_size=3, rngs=rnglib.Rngs(0))
    x_conv = np.ones((1, 5, 3), dtype=np.float32)
    assert c1(x_conv).shape == (1, 5, 4)

    c2 = nnx.Conv(
        3,
        4,
        kernel_size=(3, 3),
        strides=(1, 1),
        input_dilation=(1, 1),
        kernel_dilation=(1, 1),
        use_bias=False,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    x_conv2 = np.ones((1, 5, 5, 3), dtype=np.float32)
    c2_out = np.asarray(c2(x_conv2))
    assert c2_out.shape == (1, 5, 5, 4)
    assert float(c2_out[0, 2, 2, 0]) == 27.0

    e1 = nnx.Embed(10, 4, rngs=rnglib.Rngs(0))
    idx = np.array([1, 2, 3], dtype=np.int32)
    assert e1(idx).shape == (3, 4)

    e2 = nnx.Embed(10, 4, embedding_init=initializers.ones, promote_dtype=None, dtype="float32")
    np.testing.assert_allclose(np.asarray(e2(idx)), np.ones((3, 4), dtype=np.float32))
    np.testing.assert_allclose(
        np.asarray(e2.attend(np.ones((2, 4), dtype=np.float32))),
        np.full((2, 10), 4.0, dtype=np.float32),
    )

    mha = nnx.MultiHeadDotProductAttention(
        2, 4, 8, kernel_init=initializers.ones, bias_init=initializers.zeros
    )
    np.testing.assert_allclose(
        np.asarray(mha(np.ones((1, 2, 4), dtype=np.float32))),
        np.full((1, 2, 4), 32.0, dtype=np.float32),
        rtol=1e-5,
    )


def test_linear() -> None:
    """Verify Linear, LinearGeneral, and Einsum across sequence and integer feature configurations."""
    x = np.ones((2, 3), dtype=np.float32)
    l1 = Linear(
        3,
        4,
        rngs=rnglib.Rngs(0),
        preferred_element_type="float32",
        promote_dtype=None,
        dtype="float32",
    )
    assert l1(x).shape == (2, 4)

    l2 = Linear(
        3,
        4,
        use_bias=False,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dot_general=nnx.DotGeneralT(),
    )
    np.testing.assert_allclose(np.asarray(l2(x)), np.full((2, 4), 3.0, dtype=np.float32))

    lg1 = LinearGeneral(
        3,
        4,
        axis=-1,
        batch_axis=None,
        rngs=rnglib.Rngs(0),
        preferred_element_type="float32",
        promote_dtype=None,
        dtype="float32",
    )
    assert lg1(x).shape == (2, 4)
    assert lg1(np.ones((3,), dtype=np.float32)).shape == (4,)

    lg2 = LinearGeneral(
        (2, 3),
        (4, 5),
        axis=(-2, -1),
        use_bias=False,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dot_general_cls=nnx.DotGeneralT,
    )
    x_lg2 = np.ones((2, 2, 3), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(lg2(x_lg2)), np.full((2, 4, 5), 6.0, dtype=np.float32))

    lg_batch = LinearGeneral(
        3,
        4,
        axis=-1,
        batch_axis=[0],
        use_bias=True,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    np.testing.assert_allclose(
        np.asarray(lg_batch(np.ones((2, 3), dtype=np.float32))),
        np.full((2, 4), 3.0, dtype=np.float32),
    )

    e1 = Einsum(
        "ab,bc->ac",
        (3, 4),
        (4,),
        rngs=rnglib.Rngs(0),
        preferred_element_type="float32",
        promote_dtype=None,
        dtype="float32",
    )
    assert e1(x).shape == (2, 4)

    e2 = Einsum(
        "ab,bc",
        (3, 4),
        None,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    np.testing.assert_allclose(np.asarray(e2(x)), np.full((2, 4), 3.0, dtype=np.float32))


def test_missing_reexports() -> None:
    """Verify all compatibility re-exports on zero_flax.nnx.missing and filter composition."""
    f = filterlib.Filter()
    assert f(("a",), variables.Param(1))
    e = filterlib.Everything()
    assert e(("a",), variables.Param(1))
    n = filterlib.Nothing()
    assert not n(("a",), variables.Param(1))
    pc = filterlib.PathContains("kernel")
    assert pc(("l1", "kernel"), variables.Param(1))
    assert filterlib.PathContains("l1/kernel")(("l1", "kernel"), variables.Param(1))
    assert filterlib.PathContains("l1.kernel")(("l1", "kernel"), variables.Param(1))
    ot = filterlib.OfType(variables.Param)
    assert ot(("p",), variables.Param(1))
    a = ot & pc
    assert a(("kernel",), variables.Param(1))
    af = ot | n
    assert af(("p",), variables.Param(1))
    nt = ~n
    assert nt(("p",), variables.Param(1))
    seq_pred = filterlib.to_predicate([variables.Param, variables.BatchStat])
    assert seq_pred(("p",), variables.Param(1))

    v = variables.Variable(1)
    assert v.replace(2).value == 2
    assert v.copy().value == 1
    p = variables.Param(2)
    b = variables.BatchStat(3)
    c = variables.Cache(4)
    i = variables.Intermediate(5)
    lp = nnx.LoRAParam(6)
    assert (p.value, b.value, c.value, i.value, lp.value) == (2, 3, 4, 5, 6)

    rs = rnglib.RngState(1)
    rc = rnglib.RngCount(2)
    rk = rnglib.RngKey(nnx.Rngs(0)())
    assert rs.value == 1 and rc.value == 2 and rk.value is not None

    stream = rnglib.RngStream(0)
    assert stream() is not None
    r = rnglib.Rngs(0, params=1)
    assert r() is not None
    assert r.default() is not None
    assert r.params() is not None
    assert r.dropout() is not None
    assert r["params"] is not None

    dg = nnx.DotGeneralT()
    np.testing.assert_allclose(
        np.asarray(
            dg(
                np.ones((2, 3)),
                np.ones((3, 2)),
                (((1,), (0,)), ((), ())),
                preferred_element_type="float32",
            )
        ),
        np.full((2, 2), 3.0, dtype=np.float32),
    )
    init = nnx.Initializer()
    assert init(None, (2, 2)).shape == (2, 2)
    mf = nnx.MaxFun()
    np.testing.assert_allclose(
        np.asarray(mf(np.array([1.0, 3.0]), np.array([2.0, 1.0]))),
        np.array([2.0, 3.0], dtype=np.float32),
    )
    assert float(np.asarray(mf(np.array([1.0, 3.0])))) == 3.0


def test_module_without_super_init() -> None:
    """Verify Module subclass that omits super().__init__() still initializes registries via register_variable."""

    class RawSubModule(nnx.Module):
        """Module subclass registering a variable without calling super().__init__()."""

        p: state.Param[int]

        def __init__(self) -> None:
            """Register ``self.p = Param(1)`` without ``super().__init__()``."""
            self.p = self.register_variable("p", state.Param(1))

    m = RawSubModule()
    st = m.state()
    assert "p" in st
    m.update(state.State({"p": state.Param(99)}))
    assert m.p.value == 99

    gdef, st_split = nnx.split(m)
    m_reconstructed = nnx.merge(gdef, st_split)
    assert m_reconstructed.p.value == 99

    merged_states = nnx.merge(state.State({"a": 1}), state.State({"b": 2}))
    assert merged_states["a"] == 1 and merged_states["b"] == 2


def test_normalization_and_stochastic_variants() -> None:
    """Verify BatchNorm, LayerNorm, RMSNorm, and Dropout with custom initializers and broadcast_dims."""
    x = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)

    bn = nnx.BatchNorm(
        3,
        epsilon=0.0,
        rngs=rnglib.Rngs(0),
        scale_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    np.testing.assert_allclose(
        np.asarray(bn(x, use_running_average=False)),
        np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        atol=1e-5,
    )
    assert bn(x, use_running_average=True).shape == (2, 3)

    ln = nnx.LayerNorm(3, use_scale=False, use_bias=False)
    ln_out = np.asarray(ln(x))
    np.testing.assert_allclose(np.mean(ln_out, axis=-1), np.zeros((2,)), atol=1e-5)

    rn = nnx.RMSNorm(3, use_scale=False, epsilon=0.0)
    np.testing.assert_allclose(
        np.asarray(rn(np.full((2, 3), 4.0, dtype=np.float32))),
        np.ones((2, 3), dtype=np.float32),
        atol=1e-5,
    )

    d = nnx.Dropout(0.5, broadcast_dims=(0,), deterministic=False, rngs=rnglib.Rngs(0))
    d_out = np.asarray(d(x))
    assert d_out.shape == (2, 3)
    assert np.all((d_out == 0.0) | np.isclose(d_out, x * 2.0))
