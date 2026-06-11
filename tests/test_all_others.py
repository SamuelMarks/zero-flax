from zero_flax.nnx.state import Variable, Param, BatchStat, Rng, State, merge
from zero_flax.nnx.module import Module
from zero_jax.nn.initializers import zeros, ones, glorot_uniform, he_normal
from zero_flax.nnx import (
    Dense,
    LayerNorm,
    RMSNorm,
    Conv,
    Embed,
    MultiHeadAttention,
)
import numpy as np


def test_state():
    v = Variable(1)
    assert v.value == 1
    p = Param(2)
    b = BatchStat(3)
    Rng(4)
    s = State({"a": p, "b": b})
    assert len(s) == 2
    s1, s2, _ = s.split(Param, BatchStat)
    assert "a" in s1
    assert "b" in s2
    s3 = merge(s1, s2)
    assert "a" in s3 and "b" in s3

    # cover module
    class MyMod(Module):
        def __init__(self):
            super().__init__()
            self.p = Param(1)
            super().__setattr__("_is_initializing", False)

    m = MyMod()
    st = m.state()
    assert "p" in st
    m.update(st)


def test_initializers():
    import zero_jax as jax

    rng = jax.random.PRNGKey(0)
    shape = (2, 2)
    zeros(rng, shape)
    ones(rng, shape)
    glorot_uniform()(rng, shape)
    he_normal()(rng, shape)


def test_layers():
    x = np.ones((1, 2, 2))
    d = Dense(2, 2)
    d(x)
    d2 = Dense(2, 2, use_bias=False)
    d2(x)

    ln = LayerNorm(2)
    ln(x)
    ln(np.array([1.0]))

    rn = RMSNorm(2)
    rn(x)
    rn(np.array([1.0]))

    c = Conv(2, 2, (1, 1))
    c(x)

    e = Embed(10, 2)
    e(np.array([0, 1]))

    mha = MultiHeadAttention(2, 2)
    mha(x, x, x)
