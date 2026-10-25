"""Unit tests for `zero_flax.core` compatibility exports and `FrozenDict` utilities."""

from __future__ import annotations

import numpy as np
from zero_jax.nn import initializers

from zero_flax.core import (
    BatchNorm,
    BatchStat,
    Conv,
    ConvTranspose,
    Dense,
    Dropout,
    Embed,
    FrozenDict,
    Intermediate,
    Jit,
    LayerNorm,
    LoRA,
    LoRALinear,
    Module,
    MultiHeadAttention,
    MultiHeadDotProductAttention,
    Param,
    Pmap,
    Remat,
    RMSNorm,
    Rngs,
    Scan,
    Sequential,
    State,
    Variable,
    Vmap,
    copy,
    freeze,
    merge,
    pop,
    pretty_repr,
    split,
    state,
    unfreeze,
    update,
)


def test_core_exports() -> None:
    """Verify all symbols exported from zero_flax.core instantiate and execute accurately."""
    v = Variable(1)
    assert v.value == 1

    p = Param(2)
    assert p.value == 2

    b = BatchStat(3)
    assert b.value == 3

    i = Intermediate(4)
    assert i.value == 4

    s = State({"a": p})
    assert s["a"] == p

    r = Rngs(0)
    assert r() is not None

    m = Module()
    m.register_variable("a", Param(0))
    assert isinstance(state(m), State)
    update(m, s)
    assert m._variables["a"].value == 2
    gdef, st = split(m)
    m2 = merge(gdef, st)
    assert isinstance(m2, Module)
    assert m2._variables["a"].value == 2

    x = np.ones((2, 2), dtype=np.float32)

    d = Dense(2, 2, kernel_init=initializers.ones, bias_init=initializers.zeros)
    np.testing.assert_allclose(np.asarray(d(x)), np.full((2, 2), 2.0, dtype=np.float32))

    c = Conv(2, 2, (2, 2), kernel_init=initializers.ones, bias_init=initializers.zeros)
    x_4d = np.ones((1, 2, 2, 2), dtype=np.float32)
    c_out = np.asarray(c(x_4d))
    assert c_out.shape == (1, 2, 2, 2)
    assert float(c_out[0, 0, 0, 0]) == 8.0

    ct = ConvTranspose(2, 2, (2, 2), kernel_init=initializers.ones, bias_init=initializers.zeros)
    ct_out = np.asarray(ct(x_4d))
    assert ct_out.shape == (1, 2, 2, 2)
    assert float(ct_out[0, 0, 0, 0]) == 2.0
    assert float(ct_out[0, 1, 1, 0]) == 8.0

    e = Embed(10, 2, embedding_init=initializers.ones)
    np.testing.assert_allclose(
        np.asarray(e(np.array([0, 1], dtype=np.int32))),
        np.ones((2, 2), dtype=np.float32),
    )

    ln = LayerNorm(2)
    x_asym = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(ln(x_asym)),
        np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
        atol=1e-4,
    )

    rn = RMSNorm(2, epsilon=0.0)
    np.testing.assert_allclose(np.asarray(rn(x)), np.ones((2, 2), dtype=np.float32))

    bn = BatchNorm(2)
    np.testing.assert_allclose(
        np.asarray(bn(x_asym, use_running_average=False)),
        np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
        atol=1e-3,
    )

    dr = Dropout(0.5)
    np.testing.assert_allclose(np.asarray(dr(x, deterministic=True)), x)

    mha = MultiHeadAttention(2, 2, kernel_init=initializers.ones, bias_init=initializers.zeros)
    np.testing.assert_allclose(np.asarray(mha(x, x, x)), np.full((2, 2), 4.0, dtype=np.float32))

    mhdpa = MultiHeadDotProductAttention(
        2, 2, kernel_init=initializers.ones, bias_init=initializers.zeros
    )
    np.testing.assert_allclose(np.asarray(mhdpa(x, x, x)), np.full((2, 2), 4.0, dtype=np.float32))

    seq = Sequential(d, rn)
    np.testing.assert_allclose(np.asarray(seq(x)), np.ones((2, 2), dtype=np.float32), atol=1e-5)

    lora = LoRA(2, 2, 2, a_initializer=initializers.ones, b_initializer=initializers.ones)
    np.testing.assert_allclose(np.asarray(lora(x)), np.full((2, 2), 4.0, dtype=np.float32))

    loralin = LoRALinear(
        2,
        2,
        lora_rank=2,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        a_initializer=initializers.ones,
        b_initializer=initializers.ones,
    )
    np.testing.assert_allclose(np.asarray(loralin(x)), np.full((2, 2), 6.0, dtype=np.float32))

    def construct() -> Dense:
        """Construct a fresh Dense(2, 2) layer for transform wrapper tests.

        Returns:
            Initialized ``Dense`` instance.
        """
        return Dense(2, 2, kernel_init=initializers.ones, bias_init=initializers.zeros)

    j = Jit(construct)
    assert j.module_constructor == construct
    np.testing.assert_allclose(np.asarray(j(x)), np.full((2, 2), 2.0, dtype=np.float32))

    vm = Vmap(construct)
    assert vm.module_constructor == construct
    np.testing.assert_allclose(np.asarray(vm(x)), np.full((2, 2), 2.0, dtype=np.float32))

    sc = Scan(construct, in_axes=0, out_axes=0)
    assert sc.module_constructor == construct
    np.testing.assert_allclose(np.asarray(sc(x)), np.full((2, 2), 2.0, dtype=np.float32))

    rm = Remat(construct)
    assert rm.module_constructor == construct
    np.testing.assert_allclose(np.asarray(rm(x)), np.full((2, 2), 2.0, dtype=np.float32))

    pm = Pmap(construct)
    assert pm.module_constructor == construct
    np.testing.assert_allclose(np.asarray(pm(x)), np.full((2, 2), 2.0, dtype=np.float32))


def test_frozendict_and_helpers() -> None:
    """Verify FrozenDict, freeze, unfreeze, copy, pop, hash, and pretty_repr in zero_flax.core."""
    fd = freeze({"a": 1, "nested": {"b": 2}, "var": Param(5)})
    assert isinstance(fd, FrozenDict)
    assert freeze(fd) is fd
    assert len(fd) == 3
    assert list(fd) == ["a", "nested", "var"]
    assert fd["a"] == 1
    assert isinstance(fd["nested"], FrozenDict)
    assert fd["nested"]["b"] == 2
    assert "FrozenDict" in repr(fd)
    assert hash(fd) == hash(freeze({"a": 1, "nested": {"b": 2}, "var": Param(5)}))

    fd_iter = FrozenDict([("x", 10), ("y", {"z": 20})], w=FrozenDict({"k": 30}))
    assert fd_iter["x"] == 10
    assert isinstance(fd_iter["y"], FrozenDict)
    assert isinstance(fd_iter["w"], FrozenDict)
    assert len(FrozenDict()) == 0

    copied = copy(fd, {"c": 3, "sub": {"d": 4}})
    assert copied["c"] == 3
    assert isinstance(copied["sub"], FrozenDict)
    assert copy(fd)["a"] == 1

    popped_fd, val = pop(fd, "a")
    assert val == 1
    assert "a" not in popped_fd

    unfrozen = unfreeze(fd)
    assert unfrozen["a"] == 1 and unfrozen["nested"] == {"b": 2}
    assert unfreeze({"a": 1, "nested": {"b": 2}}) == {"a": 1, "nested": {"b": 2}}

    formatted = pretty_repr(fd, num_spaces=2)
    assert "nested:" in formatted
    assert "a: 1" in formatted
