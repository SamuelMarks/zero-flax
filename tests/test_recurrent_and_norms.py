"""Unit tests for recurrent cells, `RNN`, `Bidirectional`, `GroupNorm`, `InstanceNorm`, `SpectralNorm`, `WeightNorm`, `RoPE`, and `PReLU`."""

from __future__ import annotations

import numpy as np
import pytest

import zero_flax.nnx as nnx
from zero_flax.types import JaxArray, ensure_array


def test_recurrent_cells_and_branches() -> None:
    """Verify RNNCellBase, SimpleCell, GRUCell, LSTMCell, and OptimizedLSTMCell across all branches."""
    base = nnx.RNNCellBase(1, a=2)
    c_base = base.initialize_carry((2, 3))
    c_out, y_out = base(c_base, np.ones((2, 3), dtype=np.float32))
    assert np.asarray(c_out).shape == (2, 1)
    assert np.asarray(y_out).shape == (2, 3)

    c_tuple, y_tuple = base((ensure_array(c_base), ensure_array(c_base)), np.ones((2, 3)))
    assert isinstance(c_tuple, tuple)
    assert np.asarray(y_tuple).shape == (2, 3)

    sc = nnx.SimpleCell(3, 4, residual=True, activation_fn="relu", keep_rngs=True, rngs=nnx.Rngs(0))
    sc_carry = sc.initialize_carry((2, 3))
    sc_c2, sc_y2 = sc((sc_carry, sc_carry), np.ones((2, 3), dtype=np.float32))
    assert np.asarray(sc_c2).shape == (2, 4)
    assert np.asarray(sc_y2).shape == (2, 4)

    gru = nnx.GRUCell(
        3,
        4,
        gate_fn=nnx.sigmoid,
        activation_fn=nnx.relu,
        keep_rngs=True,
        rngs=nnx.Rngs(0),
    )
    gru_c, gru_y = gru((sc_carry, sc_carry), np.ones((2, 3), dtype=np.float32))
    assert np.asarray(gru_c).shape == (2, 4)
    assert np.asarray(gru_y).shape == (2, 4)

    lstm = nnx.LSTMCell(
        3,
        4,
        gate_fn=nnx.sigmoid,
        activation_fn=nnx.relu,
        keep_rngs=True,
        rngs=nnx.Rngs(0),
    )
    (lc, lh), ly = lstm(sc_carry, np.ones((2, 3), dtype=np.float32))
    assert np.asarray(lc).shape == (2, 4)
    assert np.asarray(lh).shape == (2, 4)
    assert np.asarray(ly).shape == (2, 4)

    opt_lstm = nnx.OptimizedLSTMCell(
        3,
        4,
        gate_fn=nnx.sigmoid,
        activation_fn=nnx.relu,
        keep_rngs=True,
        rngs=nnx.Rngs(0),
    )
    (olc, olh), oly = opt_lstm(sc_carry, np.ones((2, 3), dtype=np.float32))
    assert np.asarray(olc).shape == (2, 4)
    assert np.asarray(olh).shape == (2, 4)
    assert np.asarray(oly).shape == (2, 4)


def test_rnn_and_bidirectional_options() -> None:
    """Verify RNN and Bidirectional with time_major, reverse, keep_order, empty seq, and return_carry."""
    cell = nnx.SimpleCell(2, 3, rngs=nnx.Rngs(0))
    rnn = nnx.RNN(cell, time_major=True, reverse=True, keep_order=True, return_carry=True)
    seq_tm = np.ones((4, 2, 2), dtype=np.float32)
    carry, out = rnn(seq_tm)
    assert np.asarray(carry).shape == (2, 3)
    assert np.asarray(out).shape == (4, 2, 3)

    empty_c, empty_out = rnn(np.zeros((0, 2, 2), dtype=np.float32))
    assert np.asarray(empty_c).shape == (2, 3)
    assert np.asarray(empty_out).shape == (0, 2, 3)
    empty_only = rnn(np.zeros((2, 0, 2), dtype=np.float32), time_major=False, return_carry=False)
    assert np.asarray(empty_only).shape == (2, 0, 3)

    fwd_rnn = nnx.RNN(nnx.SimpleCell(2, 3))
    bwd_rnn = nnx.RNN(nnx.SimpleCell(2, 3), reverse=True, keep_order=True)

    def sum_merge(a: JaxArray, b: JaxArray) -> JaxArray:
        """Add forward and backward sequence outputs elementwise.

        Args:
            a: Forward sequence tensor.
            b: Backward sequence tensor.

        Returns:
            Elementwise sum ``a + b``.
        """
        return a + b

    bi = nnx.Bidirectional(
        fwd_rnn,
        bwd_rnn,
        merge_fn=sum_merge,
        return_carry=True,
    )
    init_c = ensure_array(np.zeros((2, 3), dtype=np.float32))
    (fc, bc), bi_out = bi(
        np.ones((2, 4, 2), dtype=np.float32),
        initial_carry=(init_c, init_c),
    )
    assert np.asarray(fc).shape == (2, 3)
    assert np.asarray(bc).shape == (2, 3)
    assert np.asarray(bi_out).shape == (2, 4, 3)


def test_groupnorm_and_instancenorm_branches() -> None:
    """Verify GroupNorm, InstanceNorm, BatchNorm, LayerNorm, and RMSNorm with masks and edge cases."""
    gn_gs = nnx.GroupNorm(
        num_features=6, num_groups=None, group_size=3, use_scale=False, use_bias=False
    )
    x = np.ones((2, 4, 6), dtype=np.float32)
    mask = np.ones((2, 4, 6), dtype=np.float32)
    assert gn_gs(x, mask=mask).shape == (2, 4, 6)

    gn_red = nnx.GroupNorm(num_features=4, num_groups=2, reduction_axes=(-1,))
    assert gn_red(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)

    with pytest.raises(ValueError):
        nnx.GroupNorm(num_features=5, group_size=2)

    with pytest.raises(ValueError):
        nnx.GroupNorm(num_features=5, num_groups=2)

    inorm_no = nnx.InstanceNorm(num_features=3, use_scale=False, use_bias=False)
    assert inorm_no(np.ones((2, 3), dtype=np.float32), mask=np.ones((2, 3))).shape == (2, 3)
    inorm_1d = nnx.InstanceNorm(num_features=3)
    assert inorm_1d(np.array([1.0, 2.0, 3.0], dtype=np.float32)).shape == (3,)

    bn = nnx.BatchNorm(2)
    assert bn(np.ones((2, 2), dtype=np.float32), mask=np.ones((2, 2))).shape == (2, 2)
    assert bn(np.array([1.0, 2.0], dtype=np.float32), use_running_average=True).shape == (2,)
    ln = nnx.LayerNorm(2)
    assert ln(np.ones((2, 2), dtype=np.float32), mask=np.ones((2, 2))).shape == (2, 2)
    rn = nnx.RMSNorm(2, subtract_mean=False) if False else nnx.RMSNorm(2)
    assert rn(np.ones((2, 2), dtype=np.float32), mask=np.ones((2, 2))).shape == (2, 2)

    with pytest.raises(ValueError):
        nnx.Dropout(-0.1)


def test_spectral_and_weight_norm_branches() -> None:
    """Verify SpectralNorm and WeightNorm edge cases and error branches."""
    with pytest.raises(ValueError):
        nnx.SpectralNorm(nnx.Linear(2, 2), n_steps=0)

    class OneDParamMod(nnx.Module):
        """Module with a 1D kernel parameter for testing SpectralNorm non-matrix handling."""

        kernel: nnx.Param[JaxArray]

        def __init__(self) -> None:
            """Initialize with 1D kernel."""
            super().__init__()
            self.kernel = self.register_variable("kernel", nnx.Param(ensure_array([1.0, 2.0])))
            self.finish_init()

        def __call__(self, x: JaxArray) -> JaxArray:
            """Multiply ``x`` by ``self.kernel.value``.

            Args:
                x: Input array.

            Returns:
                Scaled array.
            """
            return x * self.kernel.value

    with pytest.raises(ValueError):
        nnx.SpectralNorm(OneDParamMod(), error_on_non_matrix=True)

    sn_1d_ok = nnx.SpectralNorm(OneDParamMod(), error_on_non_matrix=False)
    assert sn_1d_ok(ensure_array([1.0, 2.0])).shape == (2,)

    class ScalarParamMod(nnx.Module):
        """Module with a 0D scalar kernel parameter for testing SpectralNorm and WeightNorm."""

        kernel: nnx.Param[JaxArray]

        def __init__(self) -> None:
            """Initialize with 0D kernel."""
            super().__init__()
            self.kernel = self.register_variable("kernel", nnx.Param(ensure_array(2.0)))
            self.finish_init()

        def __call__(self, x: JaxArray) -> JaxArray:
            """Multiply ``x`` by ``self.kernel.value``.

            Args:
                x: Input array.

            Returns:
                Scaled array.
            """
            return x * self.kernel.value

    sn_0d = nnx.SpectralNorm(ScalarParamMod(), error_on_non_matrix=False)
    assert sn_0d.u is not None
    wn_0d = nnx.WeightNorm(ScalarParamMod(), feature_axes=None)
    assert wn_0d(ensure_array([1.0, 2.0])).shape == (2,)

    empty_mod = nnx.Module()
    sn_empty = nnx.SpectralNorm(empty_mod)
    assert sn_empty(np.ones((2, 2), dtype=np.float32)).shape == (2, 2)
    wn_empty = nnx.WeightNorm(empty_mod)
    assert wn_empty(np.ones((2, 2), dtype=np.float32)).shape == (2, 2)

    sn_no_update = nnx.SpectralNorm(nnx.Linear(2, 2), update_stats=False)
    assert sn_no_update(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    wn_no_scale = nnx.WeightNorm(nnx.Linear(2, 3), use_scale=False)
    assert wn_no_scale(np.ones((1, 2), dtype=np.float32)).shape == (1, 3)

    wn_bias_filter = nnx.WeightNorm(nnx.Linear(2, 3), variable_filter=nnx.PathContains("bias"))
    assert wn_bias_filter(np.ones((1, 2), dtype=np.float32)).shape == (1, 3)


def test_rope_and_attention_branches() -> None:
    """Verify RoPE with 4D multi-head tensors, max_seq_len, odd features, and MHA validation."""
    rope = nnx.RoPE(theta=10000.0, max_seq_len=2)
    x_odd = np.ones((1, 3, 5), dtype=np.float32)
    pos = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    assert rope(x_odd, positions=pos).shape == (1, 3, 5)

    pos_2d = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    assert rope(x_odd, positions=pos_2d).shape == (1, 3, 5)
    assert rope(x_odd, positions=np.array(0.0, dtype=np.float32)).shape == (1, 3, 5)

    x_4d = np.ones((2, 3, 4, 6), dtype=np.float32)
    assert rope(x_4d).shape == (2, 3, 4, 6)

    rope_too_large = nnx.RoPE(embedding_size=10)
    assert rope_too_large(np.ones((1, 2, 4), dtype=np.float32)).shape == (1, 2, 4)

    x_1d = np.ones((4,), dtype=np.float32)
    assert rope(x_1d).shape == (4,)

    prelu_no_promote = nnx.PReLU(promote_dtype=None)
    assert prelu_no_promote(np.array([-1.0, 1.0], dtype=np.float32)).shape == (2,)

    with pytest.raises(ValueError, match="qkv_features"):
        nnx.MultiHeadAttention(num_heads=4, in_features=6, qkv_features=5)

    with pytest.raises(ValueError, match="num_kv_heads"):
        nnx.MultiHeadAttention(num_heads=4, in_features=8, num_kv_heads=3)

    mha = nnx.MultiHeadAttention(
        num_heads=4,
        in_features=6,
        qkv_features=8,
        num_kv_heads=2,
        normalize_qk=True,
    )
    mha.init_cache((2, 3, 6))
    assert mha.cached_key is not None
    assert mha.cached_value is not None
    assert mha.cache_index is not None
    assert mha(np.ones((2, 3, 6), dtype=np.float32)).shape == (2, 3, 6)

    def custom_attn(
        query: JaxArray,
        key: JaxArray,
        value: JaxArray,
        mask: nnx.Array | None = None,
        dropout_rng: JaxArray | tuple[int, int] | int | None = None,
        dropout_rate: float = 0.0,
        broadcast_dropout: bool = True,
        deterministic: bool = False,
        dtype: nnx.DType | None = None,
        precision: nnx.PrecisionLike | str | tuple[str, str] | None = None,
    ) -> JaxArray:
        """Custom attention function returning query tensor for testing.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Optional attention mask.
            dropout_rng: Optional dropout PRNG key.
            dropout_rate: Dropout probability.
            broadcast_dropout: Whether to broadcast dropout across batch dims.
            deterministic: Whether dropout is disabled.
            dtype: Optional computation dtype.
            precision: Optional numerical precision.

        Returns:
            Query tensor ``query``.
        """
        _ = (
            key,
            value,
            mask,
            dropout_rng,
            dropout_rate,
            broadcast_dropout,
            deterministic,
            dtype,
            precision,
        )
        return query

    mha_custom = nnx.MultiHeadAttention(2, 4, attention_fn=custom_attn)
    assert mha_custom(np.ones((1, 2, 4), dtype=np.float32), sow_weights=False).shape == (1, 2, 4)
    assert mha_custom(np.ones((1, 2, 4), dtype=np.float32), sow_weights=True).shape == (1, 2, 4)
    assert mha_custom.attention_weights is not None

    q3 = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.ones((2, 2, 2), dtype=bool)
    bias = np.zeros((2, 2, 2), dtype=np.float32)
    dpa = nnx.dot_product_attention(
        q3,
        np.ones((2, 1, 2), dtype=np.float32),
        np.ones((2, 1, 2), dtype=np.float32),
        bias=bias,
        mask=mask,
        broadcast_dropout=False,
        dropout_rate=0.2,
        deterministic=False,
    )
    assert dpa.shape == (2, 2, 2)
