"""Tests for the Flax (NNX) compatibility layer."""

from zero_flax.core import (
    Module,
    GraphDef,
    State,
    Variable,
    Param,
    BatchStat,
    Rng,
    Dense,
    Linear,
    LinearGeneral,
    Einsum,
    LoRA,
    LoRALinear,
    Conv,
    ConvTranspose,
    Embed,
    MultiHeadAttention,
    MultiHeadDotProductAttention,
    BatchNorm,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
    List,
    Dict,
    Jit,
    Vmap,
    Scan,
    Remat,
    Pmap,
)


def test_core_architecture() -> None:
    """Test core architecture elements."""
    m = Module(test=1)
    assert m.test == 1

    gd = GraphDef()
    assert gd is not None

    s = State({"a": 1})
    assert s["a"] == 1

    v = Variable(2)
    assert v.value == 2

    p = Param(3)
    assert p.value == 3

    b = BatchStat(4)
    assert b.value == 4

    r = Rng(5)
    assert r.value == 5


def test_linear_layers() -> None:
    """Test linear layers."""
    d = Dense(in_features=2, out_features=4)
    assert d.in_features == 2
    assert d.out_features == 4

    lin = Linear(in_features=2, out_features=4)
    assert lin.in_features == 2

    lg = LinearGeneral(in_features=2, out_features=4)
    assert lg.in_features == 2

    e = Einsum(einsum_str="ab,bc->ac", kernel_shape=(2, 3))
    assert e.einsum_str == "ab,bc->ac"

    lora = LoRA()
    assert lora is not None

    ll = LoRALinear(in_features=2, out_features=4)
    assert ll.in_features == 2


def test_conv_layers() -> None:
    """Test convolutional layers."""
    c = Conv(in_features=3, out_features=16, kernel_size=(3, 3))
    assert c.in_features == 3
    assert c.kernel_size == (3, 3)

    ct = ConvTranspose(in_features=16, out_features=3, kernel_size=(3, 3))
    assert ct.in_features == 16
    assert ct.kernel_size == (3, 3)


def test_attention_embeddings() -> None:
    """Test attention and embeddings."""
    e = Embed(num_embeddings=100, features=32)
    assert e.num_embeddings == 100
    assert e.features == 32

    mha = MultiHeadAttention(num_heads=4, qkv_features=64)
    assert mha.num_heads == 4
    assert mha.qkv_features == 64

    mhdp = MultiHeadDotProductAttention(num_heads=4, qkv_features=64)
    assert mhdp.num_heads == 4


def test_norm_layers() -> None:
    """Test normalization layers."""
    bn = BatchNorm(num_features=32)
    assert bn.num_features == 32

    ln = LayerNorm(num_features=32)
    assert ln.num_features == 32

    rn = RMSNorm(num_features=32)
    assert rn.num_features == 32


def test_stochastic_layers() -> None:
    """Test stochastic layers."""
    do = Dropout(rate=0.5)
    assert do.rate == 0.5
    assert do.rng_collection == "dropout"


def test_containers() -> None:
    """Test containers."""
    m1 = Module()
    m2 = Module()
    seq = Sequential(m1, m2)
    assert len(seq.layers) == 2

    lst = List([m1, m2])
    assert len(lst.modules) == 2

    dct = Dict({"a": m1, "b": m2})
    assert len(dct.modules) == 2


def test_transforms() -> None:
    """Test transforms."""

    def construct() -> Module:
        return Module()

    j = Jit(construct)
    assert j.module_constructor == construct

    v = Vmap(construct)
    assert v.module_constructor == construct

    s = Scan(construct)
    assert s.module_constructor == construct

    r = Remat(construct)
    assert r.module_constructor == construct

    p = Pmap(construct)
    assert p.module_constructor == construct
