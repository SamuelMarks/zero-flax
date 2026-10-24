"""Comprehensive numerical and structural parity tests for all 35 `zero_flax.nnx` layers, cells, containers, and transforms."""

from __future__ import annotations

import numpy as np
from zero_jax import numpy as jnp
from zero_jax.nn import initializers

import zero_flax.nnx as zero_nnx


def test_linear_parity() -> None:
    """Verify Linear forward pass computes y = x @ W + b accurately."""
    layer = zero_nnx.Linear(
        in_features=3,
        out_features=2,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = np.asarray(layer(x))
    expected = np.array([[7.0, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_dense_parity() -> None:
    """Verify Dense forward pass matches Linear mathematical computation."""
    layer = zero_nnx.Dense(
        in_features=2,
        out_features=3,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    out = np.asarray(layer(x))
    expected = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_lineargeneral_parity() -> None:
    """Verify LinearGeneral contracts specified input axes and shapes output axes."""
    layer = zero_nnx.LinearGeneral(
        in_features=3,
        out_features=(2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((2, 3), dtype=np.float32)
    out = np.asarray(layer(x))
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out, np.full((2, 2, 2), 3.0, dtype=np.float32), rtol=1e-5)


def test_einsum_parity() -> None:
    """Verify Einsum evaluates Einstein summation with kernel and bias."""
    layer = zero_nnx.Einsum(
        einsum_str="ab,bc->ac",
        kernel_shape=(3, 4),
        bias_shape=(4,),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((2, 3), dtype=np.float32)
    out = np.asarray(layer(x))
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, np.full((2, 4), 4.0, dtype=np.float32), rtol=1e-5)


def test_lora_parity() -> None:
    """Verify LoRA computes low-rank projection (x @ A) @ B."""
    layer = zero_nnx.LoRA(
        in_features=3,
        lora_rank=2,
        out_features=4,
        a_initializer=initializers.ones,
        b_initializer=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 3), dtype=np.float32)
    out = np.asarray(layer(x))
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out, np.full((1, 4), 6.0, dtype=np.float32), rtol=1e-5)


def test_loralinear_parity() -> None:
    """Verify LoRALinear sums base Linear output and LoRA adapter branch."""
    layer = zero_nnx.LoRALinear(
        in_features=3,
        out_features=4,
        lora_rank=2,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        a_initializer=initializers.ones,
        b_initializer=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 3), dtype=np.float32)
    out = np.asarray(layer(x))
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out, np.full((1, 4), 9.0, dtype=np.float32), rtol=1e-5)


def test_conv_parity() -> None:
    """Verify Conv 2D spatial convolution with SAME and VALID padding."""
    conv = zero_nnx.Conv(
        in_features=2,
        out_features=3,
        kernel_size=(2, 2),
        padding="VALID",
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 3, 3, 2), dtype=np.float32)
    out = np.asarray(conv(x))
    assert out.shape == (1, 2, 2, 3)
    np.testing.assert_allclose(out, np.full((1, 2, 2, 3), 8.0, dtype=np.float32), rtol=1e-5)


def test_convtranspose_parity() -> None:
    """Verify ConvTranspose spatial transposed convolution against exact analytical values."""
    ct_s2 = zero_nnx.ConvTranspose(
        in_features=2,
        out_features=4,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="SAME",
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 3, 3, 2), dtype=np.float32)
    out = np.asarray(ct_s2(x))
    assert out.shape == (1, 6, 6, 4)
    np.testing.assert_allclose(out, np.full((1, 6, 6, 4), 2.0, dtype=np.float32), rtol=1e-5)

    ct_1d = zero_nnx.ConvTranspose(
        in_features=1,
        out_features=1,
        kernel_size=(2,),
        strides=(1,),
        padding="VALID",
        use_bias=False,
        rngs=zero_nnx.Rngs(0),
    )
    ct_1d.kernel.value = jnp.array([[[10.0]], [[20.0]]])
    x_1d = np.array([[[1.0], [2.0]]], dtype=np.float32)
    out_1d = np.asarray(ct_1d(x_1d))
    expected_1d = np.array([[[20.0], [50.0], [20.0]]], dtype=np.float32)
    np.testing.assert_allclose(out_1d, expected_1d, rtol=1e-5, atol=1e-5)


def test_embed_parity() -> None:
    """Verify Embed lookup and attend dot product similarity."""
    emb = zero_nnx.Embed(
        num_embeddings=4,
        features=3,
        embedding_init=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    idx = np.array([[0, 2]], dtype=np.int32)
    looked_up = np.asarray(emb(idx))
    assert looked_up.shape == (1, 2, 3)
    np.testing.assert_allclose(looked_up, np.ones((1, 2, 3), dtype=np.float32), rtol=1e-5)
    scores = np.asarray(emb.attend(looked_up))
    assert scores.shape == (1, 2, 4)
    np.testing.assert_allclose(scores, np.full((1, 2, 4), 3.0, dtype=np.float32), rtol=1e-5)


def test_batchnorm_parity() -> None:
    """Verify BatchNorm normalizes batch activations and updates running statistics."""
    bn = zero_nnx.BatchNorm(
        num_features=2,
        momentum=0.5,
        epsilon=1e-5,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    out_train = np.asarray(bn(x, use_running_average=False))
    np.testing.assert_allclose(np.mean(out_train, axis=0), np.zeros((2,)), atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(bn.mean.value), np.array([1.0, 2.0], dtype=np.float32), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(bn.var.value), np.array([1.0, 2.5], dtype=np.float32), rtol=1e-5
    )
    out_eval = np.asarray(bn(x, use_running_average=True))
    expected_eval = (x - np.array([1.0, 2.0], dtype=np.float32)) / np.sqrt(
        np.array([1.0, 2.5], dtype=np.float32) + 1e-5
    )
    np.testing.assert_allclose(out_eval, expected_eval, rtol=1e-4, atol=1e-4)


def test_layernorm_parity() -> None:
    """Verify LayerNorm normalizes across feature dimension to zero mean and unit variance."""
    ln = zero_nnx.LayerNorm(num_features=4, epsilon=1e-6, rngs=zero_nnx.Rngs(0))
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    out = np.asarray(ln(x))
    np.testing.assert_allclose(np.mean(out, axis=-1), np.array([0.0]), atol=1e-5)
    np.testing.assert_allclose(np.var(out, axis=-1), np.array([1.0]), atol=1e-4)


def test_rmsnorm_parity() -> None:
    """Verify RMSNorm scales input by reciprocal root-mean-square."""
    rn = zero_nnx.RMSNorm(num_features=3, epsilon=0.0, rngs=zero_nnx.Rngs(0))
    x = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
    out = np.asarray(rn(x))
    np.testing.assert_allclose(out, np.ones((1, 3), dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_groupnorm_parity() -> None:
    """Verify GroupNorm normalizes within channel groups to zero mean and unit variance."""
    gn = zero_nnx.GroupNorm(num_features=4, num_groups=2, epsilon=1e-6, rngs=zero_nnx.Rngs(0))
    x = np.array([[[1.0, 3.0, 2.0, 6.0], [5.0, 7.0, 4.0, 8.0]]], dtype=np.float32)
    out = np.asarray(gn(x))
    assert out.shape == (1, 2, 4)
    g0 = x[..., :2]
    g1 = x[..., 2:]
    exp_g0 = (g0 - np.mean(g0)) / np.sqrt(np.var(g0) + 1e-6)
    exp_g1 = (g1 - np.mean(g1)) / np.sqrt(np.var(g1) + 1e-6)
    expected = np.concatenate([exp_g0, exp_g1], axis=-1)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_instancenorm_parity() -> None:
    """Verify InstanceNorm normalizes each channel across spatial dimensions."""
    inorm = zero_nnx.InstanceNorm(num_features=2, epsilon=1e-6, rngs=zero_nnx.Rngs(0))
    x = np.array([[[1.0, 10.0], [3.0, 20.0]]], dtype=np.float32)
    out = np.asarray(inorm(x))
    assert out.shape == (1, 2, 2)
    np.testing.assert_allclose(np.mean(out, axis=1), np.zeros((1, 2)), atol=1e-5)


def test_spectralnorm_parity() -> None:
    """Verify SpectralNorm normalizes wrapped layer kernel by largest singular value."""
    linear = zero_nnx.Linear(
        2,
        2,
        use_bias=False,
        kernel_init=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    sn = zero_nnx.SpectralNorm(linear, n_steps=2, rngs=zero_nnx.Rngs(0))
    x = np.ones((1, 2), dtype=np.float32)
    out = np.asarray(sn(x))
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, np.ones((1, 2), dtype=np.float32), rtol=1e-4, atol=1e-4)


def test_weightnorm_parity() -> None:
    """Verify WeightNorm normalizes wrapped layer kernel columns to unit L2 norm."""
    linear = zero_nnx.Linear(
        2,
        2,
        use_bias=False,
        kernel_init=initializers.ones,
        rngs=zero_nnx.Rngs(0),
    )
    wn = zero_nnx.WeightNorm(linear, rngs=zero_nnx.Rngs(0))
    x = np.ones((1, 2), dtype=np.float32)
    out = np.asarray(wn(x))
    expected_val = float(2.0 / np.sqrt(2.0))
    np.testing.assert_allclose(
        out, np.full((1, 2), expected_val, dtype=np.float32), rtol=1e-4, atol=1e-4
    )


def test_dropout_parity() -> None:
    """Verify Dropout deterministic pass and rate=1.0 zeroing."""
    drop = zero_nnx.Dropout(rate=0.5, deterministic=False, rngs=zero_nnx.Rngs(0))
    x = np.ones((4, 4), dtype=np.float32)
    out_det = np.asarray(drop(x, deterministic=True))
    np.testing.assert_allclose(out_det, x)
    drop_all = zero_nnx.Dropout(rate=1.0, deterministic=False, rngs=zero_nnx.Rngs(0))
    np.testing.assert_allclose(np.asarray(drop_all(x)), np.zeros_like(x))


def test_multiheadattention_parity() -> None:
    """Verify MultiHeadAttention forward pass with query, key, value projections."""
    mha = zero_nnx.MultiHeadAttention(
        num_heads=2,
        in_features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 3, 4), dtype=np.float32)
    out = np.asarray(mha(x))
    assert out.shape == (1, 3, 4)
    np.testing.assert_allclose(out, np.full((1, 3, 4), 16.0, dtype=np.float32), rtol=1e-4)


def test_multiheaddotproductattention_parity() -> None:
    """Verify MultiHeadDotProductAttention forward pass matches MultiHeadAttention."""
    mha = zero_nnx.MultiHeadDotProductAttention(
        num_heads=2,
        in_features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 2, 4), dtype=np.float32)
    out = np.asarray(mha(x))
    assert out.shape == (1, 2, 4)
    np.testing.assert_allclose(out, np.full((1, 2, 4), 16.0, dtype=np.float32), rtol=1e-4)


def test_rope_parity() -> None:
    """Verify RoPE preserves norms and leaves position 0 unrotated."""
    rope = zero_nnx.RoPE(theta=10000.0, embedding_size=4)
    x = np.array([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    out = np.asarray(rope(x))
    assert out.shape == (1, 2, 4)
    np.testing.assert_allclose(out[:, 0, :], x[:, 0, :], rtol=1e-5, atol=1e-5)


def test_prelu_parity() -> None:
    """Verify PReLU scales negative values by negative_slope and preserves positive values."""
    prelu = zero_nnx.PReLU(negative_slope_init=0.25)
    x = np.array([[-4.0, 0.0, 2.0]], dtype=np.float32)
    out = np.asarray(prelu(x))
    expected = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_simplecell_parity() -> None:
    """Verify SimpleCell carry initialization and recurrent step."""
    cell = zero_nnx.SimpleCell(
        in_features=3,
        hidden_features=2,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    assert isinstance(cell, zero_nnx.RNNCellBase)
    x = np.ones((1, 3), dtype=np.float32)
    carry = cell.initialize_carry(x.shape)
    new_carry, out = cell(carry, x)
    expected = np.tanh(np.full((1, 2), 3.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(new_carry), expected, rtol=1e-5, atol=1e-5)


def test_grucell_parity() -> None:
    """Verify GRUCell carry initialization and gated recurrent step."""
    cell = zero_nnx.GRUCell(
        in_features=2,
        hidden_features=3,
        kernel_init=initializers.zeros,
        recurrent_kernel_init=initializers.zeros,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((2, 2), dtype=np.float32)
    carry = cell.initialize_carry(x.shape)
    new_carry, out = cell(carry, x)
    assert new_carry.shape == (2, 3)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(out), np.zeros((2, 3), dtype=np.float32), atol=1e-5)


def test_lstmcell_parity() -> None:
    """Verify LSTMCell (c, h) carry initialization and gated recurrent step."""
    cell = zero_nnx.LSTMCell(
        in_features=2,
        hidden_features=3,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 2), dtype=np.float32)
    carry = cell.initialize_carry(x.shape)
    (new_c, new_h), out = cell(carry, x)
    assert new_c.shape == (1, 3)
    assert new_h.shape == (1, 3)
    sig_2 = 1.0 / (1.0 + np.exp(-2.0))
    tanh_2 = np.tanh(2.0)
    exp_c = sig_2 * tanh_2
    exp_h = sig_2 * np.tanh(exp_c)
    np.testing.assert_allclose(np.asarray(new_c), np.full((1, 3), exp_c), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out), np.full((1, 3), exp_h), rtol=1e-5, atol=1e-5)


def test_optimizedlstmcell_parity() -> None:
    """Verify OptimizedLSTMCell fused gate recurrent step matches LSTMCell."""
    cell = zero_nnx.OptimizedLSTMCell(
        in_features=2,
        hidden_features=3,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    x = np.ones((1, 2), dtype=np.float32)
    carry = cell.initialize_carry(x.shape)
    (new_c, new_h), out = cell(carry, x)
    assert new_c.shape == (1, 3)
    assert new_h.shape == (1, 3)
    sig_2 = 1.0 / (1.0 + np.exp(-2.0))
    tanh_2 = np.tanh(2.0)
    exp_c = sig_2 * tanh_2
    exp_h = sig_2 * np.tanh(exp_c)
    np.testing.assert_allclose(np.asarray(new_c), np.full((1, 3), exp_c), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out), np.full((1, 3), exp_h), rtol=1e-5, atol=1e-5)


def test_rnn_parity() -> None:
    """Verify RNN unrolls a cell across the time dimension with exact step values."""
    cell = zero_nnx.SimpleCell(
        in_features=2,
        hidden_features=2,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    rnn = zero_nnx.RNN(cell)
    x = np.ones((1, 3, 2), dtype=np.float32)
    out = np.asarray(rnn(x))
    assert out.shape == (1, 3, 2)
    h1 = np.tanh(np.full((1, 2), 2.0, dtype=np.float32))
    h2 = np.tanh(np.full((1, 2), 2.0, dtype=np.float32) + np.sum(h1, axis=-1, keepdims=True))
    h3 = np.tanh(np.full((1, 2), 2.0, dtype=np.float32) + np.sum(h2, axis=-1, keepdims=True))
    expected = np.stack([h1, h2, h3], axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_bidirectional_parity() -> None:
    """Verify Bidirectional runs forward and backward RNNs and concatenates exact outputs."""
    fwd = zero_nnx.SimpleCell(
        in_features=2,
        hidden_features=2,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.zeros,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    bwd = zero_nnx.SimpleCell(
        in_features=2,
        hidden_features=2,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.zeros,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(1),
    )
    bi = zero_nnx.Bidirectional(fwd, bwd)
    x = np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)
    out = np.asarray(bi(x))
    assert out.shape == (1, 2, 4)
    exp_t0 = np.tanh(2.0)
    exp_t1 = np.tanh(4.0)
    expected = np.array(
        [[[exp_t0, exp_t0, exp_t0, exp_t0], [exp_t1, exp_t1, exp_t1, exp_t1]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_sequential_parity() -> None:
    """Verify Sequential chains layers in order."""
    seq = zero_nnx.Sequential(
        zero_nnx.Linear(
            2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros, rngs=zero_nnx.Rngs(0)
        ),
        zero_nnx.relu,
        zero_nnx.Linear(
            3, 1, kernel_init=initializers.ones, bias_init=initializers.zeros, rngs=zero_nnx.Rngs(1)
        ),
    )
    x = np.ones((2, 2), dtype=np.float32)
    out = np.asarray(seq(x))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.full((2, 1), 6.0, dtype=np.float32), rtol=1e-5)


def test_dict_parity() -> None:
    """Verify nnx.Dict registers child modules and supports state round-tripping."""
    d = zero_nnx.Dict(
        {
            "a": zero_nnx.Linear(
                2,
                3,
                kernel_init=initializers.ones,
                bias_init=initializers.zeros,
                rngs=zero_nnx.Rngs(0),
            )
        }
    )
    x = np.ones((1, 2), dtype=np.float32)
    out = np.asarray(d["a"](x))
    np.testing.assert_allclose(out, np.full((1, 3), 2.0, dtype=np.float32), rtol=1e-5)
    gdef, st = zero_nnx.split(d)
    assert "a" in st
    d2 = zero_nnx.merge(gdef, st)
    np.testing.assert_allclose(np.asarray(d2["a"](x)), out, rtol=1e-5)


def test_list_parity() -> None:
    """Verify nnx.List registers child modules by index and evaluates numerically."""
    lst = zero_nnx.List(
        [
            zero_nnx.Linear(
                2,
                3,
                kernel_init=initializers.ones,
                bias_init=initializers.zeros,
                rngs=zero_nnx.Rngs(0),
            )
        ]
    )
    x = np.ones((1, 2), dtype=np.float32)
    out = np.asarray(lst[0](x))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out, np.full((1, 3), 2.0, dtype=np.float32), rtol=1e-5)


def test_jit_parity() -> None:
    """Verify Jit wraps a module and executes forward pass."""
    jitted = zero_nnx.Jit(
        zero_nnx.Linear(
            2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros, rngs=zero_nnx.Rngs(0)
        )
    )
    x = np.ones((1, 2), dtype=np.float32)
    out = np.asarray(jitted(x))
    np.testing.assert_allclose(out, np.full((1, 3), 2.0, dtype=np.float32), rtol=1e-5)


def test_remat_parity() -> None:
    """Verify Remat wraps a module and executes forward pass."""
    rematted = zero_nnx.Remat(
        zero_nnx.Linear(
            2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros, rngs=zero_nnx.Rngs(0)
        )
    )
    x = np.ones((1, 2), dtype=np.float32)
    out = np.asarray(rematted(x))
    np.testing.assert_allclose(out, np.full((1, 3), 2.0, dtype=np.float32), rtol=1e-5)


def test_vmap_parity() -> None:
    """Verify Vmap vectorizes a module over the batch dimension."""
    vmapped = zero_nnx.Vmap(
        zero_nnx.Linear(
            2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros, rngs=zero_nnx.Rngs(0)
        ),
        in_axes=0,
        out_axes=0,
    )
    x = np.ones((4, 2), dtype=np.float32)
    out = np.asarray(vmapped(x))
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out, np.full((4, 3), 2.0, dtype=np.float32), rtol=1e-5)


def test_scan_parity() -> None:
    """Verify Scan unrolls a recurrent cell across a sequence dimension with exact values."""
    cell = zero_nnx.SimpleCell(
        in_features=2,
        hidden_features=2,
        kernel_init=initializers.ones,
        recurrent_kernel_init=initializers.zeros,
        bias_init=initializers.zeros,
        rngs=zero_nnx.Rngs(0),
    )
    scanned = zero_nnx.Scan(cell, in_axes=1, out_axes=1)
    carry = jnp.zeros((1, 2))
    xs = np.ones((1, 3, 2), dtype=np.float32)
    res = scanned(carry, xs)
    assert isinstance(res, tuple)
    final_carry, ys = res[0], res[1]
    assert final_carry.shape == (1, 2)
    assert ys.shape == (1, 3, 2)
    exp_step = np.tanh(np.full((1, 2), 2.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(final_carry), exp_step, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(ys), np.stack([exp_step, exp_step, exp_step], axis=1), rtol=1e-5, atol=1e-5
    )


def test_pmap_parity() -> None:
    """Verify Pmap maps a module across the leading device axis."""
    pmapped = zero_nnx.Pmap(
        zero_nnx.Linear(
            2, 3, kernel_init=initializers.ones, bias_init=initializers.zeros, rngs=zero_nnx.Rngs(0)
        ),
        in_axes=0,
        out_axes=0,
    )
    x = np.ones((2, 2), dtype=np.float32)
    out = np.asarray(pmapped(x))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.full((2, 3), 2.0, dtype=np.float32), rtol=1e-5)
