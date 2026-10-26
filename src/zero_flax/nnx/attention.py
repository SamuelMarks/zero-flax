"""Multi-head attention, Rotary Positional Embedding (`RoPE`), and `PReLU` for zero-flax NNX.

This module implements `MultiHeadAttention`, `MultiHeadDotProductAttention`,
`dot_product_attention`, `RoPE`, and `PReLU` with exact `flax_nnx_v0.12.10`
parameter signatures and mathematical forward-pass execution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional, Union

from zero_jax import nn as jnn
from zero_jax import numpy as jnp
from zero_jax import random as jrandom

from zero_flax.nnx.linear import LinearGeneral
from zero_flax.nnx.module import Module
from zero_flax.nnx.normalization import LayerNorm
from zero_flax.nnx.rnglib import Rngs, RngStream, resolve_rng_key
from zero_flax.nnx.variables import Cache, Intermediate, Param
from zero_flax.types import (
    EMPTY_METADATA,
    ArrayLike,
    AttentionFnT,
    DotGeneralLike,
    DotGeneralT,
    DType,
    InitializerLike,
    JaxArray,
    MetadataMapping,
    PrecisionSpec,
    PRNGKeyLike,
    PromoteDtypeFn,
    default_bias_init,
    default_kernel_init,
    default_promote_dtype,
    ensure_array,
    ones_init,
    sanitize_dtype_str,
)


def _assert_axis_dim(tensor: JaxArray, axis: int, expected: int) -> None:
    """Validate that ``tensor`` has size ``expected`` along ``axis`` via ``zero_chex``.

    Args:
        tensor: Input ``JaxArray`` to validate.
        axis: Axis index to check.
        expected: Expected dimension size along ``axis``.
    """
    from zero_chex.assertions import assert_axis_dimension

    assert_axis_dimension(tensor, axis, expected)


def _compute_attention_weights(
    q_b: JaxArray,
    k_b: JaxArray,
    bias: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKeyLike = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> JaxArray:
    """Compute normalized (and optionally dropped-out) attention weights.

    Args:
        q_b: Batched query tensor of shape ``(*batch, q_len, num_heads, head_dim)``.
        k_b: Batched key tensor of shape ``(*batch, kv_len, num_heads, head_dim)``.
        bias: Optional additive attention bias.
        mask: Optional boolean attention mask.
        broadcast_dropout: Whether to broadcast dropout across batch and head dimensions.
        dropout_rng: Optional PRNG key for attention dropout.
        dropout_rate: Attention dropout probability.
        deterministic: Whether dropout is disabled.
        scale: Optional explicit scaling factor for queries (defaults to ``1 / sqrt(head_dim)``).
        is_causal: Whether to apply a lower-triangular causal attention mask.

    Returns:
        Attention weight tensor of shape ``(*batch, num_heads, q_len, kv_len)``.
    """
    head_dim = max(1, int(q_b.shape[-1]))
    eff_scale = float(scale) if scale is not None else (1.0 / math.sqrt(float(head_dim)))
    q_scaled = q_b * eff_scale

    logits: JaxArray = jnp.einsum("...qhd,...khd->...hqk", q_scaled, k_b)
    if bias is not None:
        logits = logits + ensure_array(bias)
    if is_causal:
        q_len = int(q_b.shape[-3])
        kv_len = int(k_b.shape[-3])
        q_idx = jnp.reshape(jnp.arange(q_len), (q_len, 1))
        k_idx = jnp.reshape(jnp.arange(kv_len), (1, kv_len))
        causal_mask = k_idx <= q_idx
        logits = jnp.where(causal_mask, logits, -1e9)
    if mask is not None:
        mask_arr = ensure_array(mask)
        logits = jnp.where(mask_arr, logits, -1e9)

    weights: JaxArray = jnn.softmax(logits, axis=-1)
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - float(dropout_rate)
        rng = jrandom.PRNGKey(0) if dropout_rng is None else ensure_array(dropout_rng)
        if broadcast_dropout and weights.ndim >= 3:
            mask_shape = (1,) * (weights.ndim - 2) + (weights.shape[-2], weights.shape[-1])
        else:
            mask_shape = weights.shape
        keep_mask = jrandom.bernoulli(rng, p=keep_prob, shape=mask_shape)
        weights = jnp.where(keep_mask, weights / keep_prob, 0.0)
    return weights


def dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    bias: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKeyLike = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[DType] = None,
    precision: PrecisionSpec = None,
    module: Optional[Module] = None,
    *,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> JaxArray:
    r"""Compute scaled dot-product attention ``softmax(Q K^T / sqrt(d_k) + B) V``.

    Args:
        query: Query tensor of shape ``(*batch, q_len, num_heads, head_dim)`` or ``(q_len, num_heads, head_dim)``.
        key: Key tensor of shape ``(*batch, kv_len, num_kv_heads, head_dim)`` or ``(kv_len, num_kv_heads, head_dim)``.
        value: Value tensor of shape ``(*batch, kv_len, num_kv_heads, head_dim)`` or ``(kv_len, num_kv_heads, head_dim)``.
        bias: Optional additive attention bias broadcastable to attention logits.
        mask: Optional boolean attention mask where ``False`` entries are masked with ``-1e9``.
        broadcast_dropout: Whether to broadcast the dropout mask across batch and head dimensions.
        dropout_rng: Optional PRNG key used when ``dropout_rate > 0.0`` and ``not deterministic``.
        dropout_rate: Dropout probability in ``[0.0, 1.0)`` applied to attention weights.
        deterministic: Whether to disable attention dropout.
        dtype: Optional computation data type.
        precision: Optional contraction precision specification.
        module: Optional calling ``Module`` instance in which to sow ``attention_weights``.
        scale: Optional explicit scaling factor for queries (defaults to ``1 / sqrt(head_dim)``).
        is_causal: Whether to apply a lower-triangular causal attention mask.

    Returns:
        Attended value ``JaxArray`` of the same shape as ``query``.
    """
    _ = precision
    q = ensure_array(query, dtype=dtype)
    k = ensure_array(key, dtype=dtype)
    v = ensure_array(value, dtype=dtype)

    if q.ndim == 3:
        q_b = jnp.expand_dims(q, axis=0)
        k_b = jnp.expand_dims(k, axis=0)
        v_b = jnp.expand_dims(v, axis=0)
        squeeze_batch = True
    else:
        q_b = q
        k_b = k
        v_b = v
        squeeze_batch = False

    num_q_heads = q_b.shape[-2]
    num_kv_heads = k_b.shape[-2]
    if num_q_heads != num_kv_heads and num_kv_heads > 0 and num_q_heads % num_kv_heads == 0:
        repeats = num_q_heads // num_kv_heads
        k_b = jnp.concatenate([k_b for _ in range(repeats)], axis=-2)
        v_b = jnp.concatenate([v_b for _ in range(repeats)], axis=-2)

    weights = _compute_attention_weights(
        q_b,
        k_b,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        scale=scale,
        is_causal=is_causal,
    )
    if module is not None:
        sown = weights[0] if squeeze_batch else weights
        module.register_variable("attention_weights", Intermediate(sown))

    attended: JaxArray = jnp.einsum("...hqk,...khd->...qhd", weights, v_b)
    if squeeze_batch:
        return attended[0]
    return attended


class MultiHeadAttention(Module):
    r"""Multi-Head Dot-Product Attention module supporting grouped-query attention and QK norm.

    Attributes:
        num_heads: Number of query attention heads.
        in_features: Input query feature dimension.
        qkv_features: Total internal projection dimension across all heads.
        out_features: Output feature dimension.
        in_kv_features: Input key/value feature dimension.
        num_kv_heads: Number of key/value heads (for Grouped-Query Attention).
        head_dim: Feature dimension per head (``qkv_features // num_heads``).
        query: ``LinearGeneral`` projection for queries.
        key: ``LinearGeneral`` projection for keys.
        value: ``LinearGeneral`` projection for values.
        out: ``LinearGeneral`` output projection.
        query_ln: Optional ``LayerNorm`` applied to projected queries when ``normalize_qk=True``.
        key_ln: Optional ``LayerNorm`` applied to projected keys when ``normalize_qk=True``.
    """

    num_heads: int
    in_features: int
    qkv_features: int
    out_features: int
    in_kv_features: int
    num_kv_heads: int
    head_dim: int
    dtype: Optional[DType]
    param_dtype: DType
    broadcast_dropout: bool
    dropout_rate: float
    deterministic: Optional[bool]
    precision: PrecisionSpec
    use_bias: bool
    attention_fn: AttentionFnT
    decode: Optional[bool]
    normalize_qk: bool
    keep_rngs: bool
    rngs: Optional[Union[Rngs, RngStream]]
    query: LinearGeneral
    key: LinearGeneral
    value: LinearGeneral
    out: LinearGeneral
    query_ln: Optional[LayerNorm]
    key_ln: Optional[LayerNorm]
    cached_key: Optional[Cache[JaxArray]]
    cached_value: Optional[Cache[JaxArray]]
    cache_index: Optional[Cache[int]]
    attention_weights: Optional[Intermediate[JaxArray]]

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: Optional[int] = None,
        out_features: Optional[int] = None,
        in_kv_features: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        *,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: Optional[bool] = None,
        precision: PrecisionSpec = None,
        kernel_init: InitializerLike = default_kernel_init,
        out_kernel_init: InitializerLike = None,
        bias_init: InitializerLike = default_bias_init,
        out_bias_init: InitializerLike = None,
        use_bias: bool = True,
        attention_fn: AttentionFnT = None,
        decode: Optional[bool] = None,
        normalize_qk: bool = False,
        qkv_promote_dtype: PromoteDtypeFn = default_promote_dtype,
        out_promote_dtype: PromoteDtypeFn = default_promote_dtype,
        ln_promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        keep_rngs: bool = True,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        out_kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        out_bias_metadata: MetadataMapping = EMPTY_METADATA,
        query_ln_scale_metadata: MetadataMapping = EMPTY_METADATA,
        key_ln_scale_metadata: MetadataMapping = EMPTY_METADATA,
        qkv_dot_general: DotGeneralLike = None,
        qkv_dot_general_cls: Optional[type[DotGeneralT]] = None,
        out_dot_general_cls: Optional[type[DotGeneralT]] = None,
        out_dot_general: DotGeneralLike = None,
    ) -> None:
        r"""Initialize a MultiHeadAttention layer.

        Args:
            num_heads: Number of query attention heads.
            in_features: Input feature dimension for queries.
            qkv_features: Total internal projection dimension across heads; defaults to ``in_features``.
            out_features: Output feature dimension; defaults to ``in_features``.
            in_kv_features: Input feature dimension for keys and values; defaults to ``in_features``.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            broadcast_dropout: Whether to broadcast attention dropout masks across batches.
            dropout_rate: Attention weight dropout rate.
            deterministic: Default flag controlling whether attention dropout is disabled.
            precision: Numerical precision specification.
            kernel_init: Initializer for Q/K/V projection weights.
            out_kernel_init: Optional initializer override for the output projection weights.
            bias_init: Initializer for Q/K/V projection biases.
            out_bias_init: Optional initializer override for the output projection bias.
            use_bias: Whether Q/K/V and output projections include bias parameters.
            attention_fn: Attention computation callable (defaults to ``dot_product_attention``).
            decode: Whether to operate in autoregressive single-step decoding mode by default.
            normalize_qk: Whether to apply LayerNorm to projected queries and keys.
            qkv_promote_dtype: Dtype promotion callable for Q/K/V projections.
            out_promote_dtype: Dtype promotion callable for the output projection.
            ln_promote_dtype: Dtype promotion callable for Q/K LayerNorms.
            rngs: Random number generator stream(s).
            keep_rngs: Whether to store the ``rngs`` stream for runtime dropout.
            kernel_metadata: Metadata mapping for Q/K/V kernels.
            out_kernel_metadata: Metadata mapping for the output kernel.
            bias_metadata: Metadata mapping for Q/K/V biases.
            out_bias_metadata: Metadata mapping for the output bias.
            query_ln_scale_metadata: Metadata mapping for ``query_ln`` scale.
            key_ln_scale_metadata: Metadata mapping for ``key_ln`` scale.
            num_kv_heads: Optional number of key/value heads (defaults to ``num_heads``).
            qkv_dot_general: Optional ``dot_general`` callable for Q/K/V projections.
            qkv_dot_general_cls: Optional ``DotGeneralT`` class for Q/K/V projections.
            out_dot_general_cls: Optional ``DotGeneralT`` class for the output projection.
            out_dot_general: Optional ``dot_general`` callable for the output projection.

        Raises:
            ValueError: If ``qkv_features`` is not divisible by ``num_heads``, or if
                ``num_heads`` is not divisible by ``num_kv_heads``.
        """
        super().__init__()
        self.num_heads = int(num_heads)
        self.in_features = int(in_features)
        resolved_qkv = self.in_features if qkv_features is None else int(qkv_features)
        if resolved_qkv <= 0 or resolved_qkv % self.num_heads != 0:
            raise ValueError(
                f"qkv_features ({resolved_qkv}) must be a positive multiple of "
                f"num_heads ({self.num_heads})."
            )
        self.qkv_features = resolved_qkv
        self.out_features = self.in_features if out_features is None else int(out_features)
        self.in_kv_features = self.in_features if in_kv_features is None else int(in_kv_features)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.num_heads
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        self.head_dim = self.qkv_features // self.num_heads

        self.dtype = dtype
        self.param_dtype = param_dtype
        self.broadcast_dropout = bool(broadcast_dropout)
        self.dropout_rate = float(dropout_rate)
        self.deterministic = deterministic
        self.precision = precision
        self.use_bias = bool(use_bias)
        self.attention_fn = attention_fn
        self.decode = decode
        self.normalize_qk = bool(normalize_qk)
        self.keep_rngs = bool(keep_rngs)
        if self.keep_rngs and isinstance(rngs, (Rngs, RngStream)):
            self.rngs = self.register_child("rngs", rngs)
        else:
            self.rngs = None

        init_rngs: Union[Rngs, RngStream] = rngs if rngs is not None else Rngs(0)
        eff_out_k_init = kernel_init if out_kernel_init is None else out_kernel_init
        eff_out_b_init = bias_init if out_bias_init is None else out_bias_init

        self.query = self.register_child(
            "query",
            LinearGeneral(
                self.in_features,
                (self.num_heads, self.head_dim),
                use_bias=self.use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                precision=precision,
                promote_dtype=qkv_promote_dtype,
                dot_general=qkv_dot_general,
                dot_general_cls=qkv_dot_general_cls,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.key = self.register_child(
            "key",
            LinearGeneral(
                self.in_kv_features,
                (self.num_kv_heads, self.head_dim),
                use_bias=self.use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                precision=precision,
                promote_dtype=qkv_promote_dtype,
                dot_general=qkv_dot_general,
                dot_general_cls=qkv_dot_general_cls,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.value = self.register_child(
            "value",
            LinearGeneral(
                self.in_kv_features,
                (self.num_kv_heads, self.head_dim),
                use_bias=self.use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                precision=precision,
                promote_dtype=qkv_promote_dtype,
                dot_general=qkv_dot_general,
                dot_general_cls=qkv_dot_general_cls,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.out = self.register_child(
            "out",
            LinearGeneral(
                (self.num_heads, self.head_dim),
                self.out_features,
                axis=(-2, -1),
                use_bias=self.use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=eff_out_k_init,
                bias_init=eff_out_b_init,
                precision=precision,
                promote_dtype=out_promote_dtype,
                dot_general=out_dot_general,
                dot_general_cls=out_dot_general_cls,
                rngs=init_rngs,
                kernel_metadata=out_kernel_metadata,
                bias_metadata=out_bias_metadata,
            ),
        )

        if self.normalize_qk:
            self.query_ln = self.register_child(
                "query_ln",
                LayerNorm(
                    self.head_dim,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    use_bias=False,
                    promote_dtype=ln_promote_dtype,
                    rngs=init_rngs,
                    scale_metadata=query_ln_scale_metadata,
                ),
            )
            self.key_ln = self.register_child(
                "key_ln",
                LayerNorm(
                    self.head_dim,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    use_bias=False,
                    promote_dtype=ln_promote_dtype,
                    rngs=init_rngs,
                    scale_metadata=key_ln_scale_metadata,
                ),
            )
        else:
            self.query_ln = None
            self.key_ln = None

        self.cached_key = None
        self.cached_value = None
        self.cache_index = None
        self.attention_weights = None
        self.finish_init()

    def init_cache(self, input_shape: Sequence[int], dtype: DType = "float32") -> None:
        r"""Initialize autoregressive decoding cache variables for ``input_shape``.

        Args:
            input_shape: Sequence ``(*batch_dims, max_length, in_features)``.
            dtype: Data type for cached key and value tensors.

        Raises:
            ValueError: If ``input_shape`` has fewer than 2 dimensions or its last
                dimension does not match ``self.in_features``.
        """
        if len(input_shape) < 2 or int(input_shape[-1]) != self.in_features:
            raise ValueError(
                f"init_cache expects input_shape (*batch, max_length, {self.in_features}), "
                f"got {tuple(input_shape)}."
            )
        cache_shape = (*tuple(int(d) for d in input_shape[:-1]), self.num_kv_heads, self.head_dim)
        self.cached_key = self.register_variable(
            "cached_key",
            Cache(ensure_array(jnp.zeros(cache_shape), dtype=dtype)),
        )
        self.cached_value = self.register_variable(
            "cached_value",
            Cache(ensure_array(jnp.zeros(cache_shape), dtype=dtype)),
        )
        self.cache_index = self.register_variable("cache_index", Cache(0))

    def __call__(
        self,
        inputs_q: Optional[ArrayLike] = None,
        inputs_k: Optional[ArrayLike] = None,
        inputs_v: Optional[ArrayLike] = None,
        *,
        mask: Optional[ArrayLike] = None,
        deterministic: Optional[bool] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        sow_weights: bool = False,
        decode: Optional[bool] = None,
    ) -> JaxArray:
        r"""Compute multi-head scaled dot-product attention over ``(inputs_q, inputs_k, inputs_v)``.

        Args:
            inputs_q: Query input tensor of shape ``(*batch, q_len, in_features)`` or ``(batch, in_features)``.
            inputs_k: Optional key input tensor (defaults to ``inputs_q`` for self-attention).
            inputs_v: Optional value input tensor (defaults to ``inputs_k``).
            mask: Optional boolean attention mask.
            deterministic: Optional override for ``self.deterministic``.
            rngs: Optional RNG stream(s) for attention dropout.
            sow_weights: Whether to record intermediate attention weights in ``self.attention_weights``.
            decode: Optional override for ``self.decode``.

        Returns:
            Output ``JaxArray`` of shape ``(*inputs_q.shape[:-1], out_features)``.

        Raises:
            ValueError: If ``inputs_q`` is ``None``, if input feature dimensions do not match
                ``in_features`` / ``in_kv_features``, if ``decode=True`` is used before
                ``init_cache()`` has been called, or if the decoding cache overflows ``max_length``.
        """
        if inputs_q is None:
            raise ValueError("MultiHeadAttention requires a non-None inputs_q tensor.")
        q_in = ensure_array(inputs_q)
        k_in = q_in if inputs_k is None else ensure_array(inputs_k)
        v_in = k_in if inputs_v is None else ensure_array(inputs_v)

        if q_in.ndim == 0 or q_in.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected inputs_q with last dimension {self.in_features}, got shape {q_in.shape}."
            )
        if k_in.ndim == 0 or k_in.shape[-1] != self.in_kv_features:
            raise ValueError(
                f"Expected inputs_k with last dimension {self.in_kv_features}, got shape {k_in.shape}."
            )
        if v_in.ndim == 0 or v_in.shape[-1] != self.in_kv_features:
            raise ValueError(
                f"Expected inputs_v with last dimension {self.in_kv_features}, got shape {v_in.shape}."
            )
        _assert_axis_dim(q_in, -1, self.in_features)
        _assert_axis_dim(k_in, -1, self.in_kv_features)
        _assert_axis_dim(v_in, -1, self.in_kv_features)

        orig_ndim = q_in.ndim
        added_seq_dim = False
        if orig_ndim == 2:
            q_in = jnp.expand_dims(q_in, axis=1)
            k_in = jnp.expand_dims(k_in, axis=1)
            v_in = jnp.expand_dims(v_in, axis=1)
            added_seq_dim = True
        elif orig_ndim == 1:
            q_in = jnp.reshape(q_in, (1, 1, self.in_features))
            k_in = jnp.reshape(k_in, (1, 1, self.in_kv_features))
            v_in = jnp.reshape(v_in, (1, 1, self.in_kv_features))
            added_seq_dim = True

        q_proj = self.query(q_in)
        k_proj = self.key(k_in)
        v_proj = self.value(v_in)

        if self.normalize_qk and self.query_ln is not None and self.key_ln is not None:
            q_proj = self.query_ln(q_proj)
            k_proj = self.key_ln(k_proj)

        do_decode = bool(decode if decode is not None else self.decode)
        if do_decode:
            if self.cached_key is None or self.cached_value is None or self.cache_index is None:
                raise ValueError(
                    "Autoregressive decode=True requires init_cache() to be called before __call__."
                )
            cur_idx = int(self.cache_index.value)
            step_len = k_proj.shape[-3]
            max_len = self.cached_key.value.shape[-3]
            after_idx = cur_idx + step_len
            if after_idx > max_len:
                raise ValueError(
                    f"Autoregressive decoding cache overflow: cache_index ({cur_idx}) + "
                    f"step_len ({step_len}) exceeds max_len ({max_len})."
                )
            cur_k_cache = ensure_array(self.cached_key.value)
            cur_v_cache = ensure_array(self.cached_value.value)
            k_parts: list[JaxArray] = []
            v_parts: list[JaxArray] = []
            if cur_idx > 0:
                k_parts.append(cur_k_cache[..., :cur_idx, :, :])
                v_parts.append(cur_v_cache[..., :cur_idx, :, :])
            k_parts.append(k_proj)
            v_parts.append(v_proj)
            if after_idx < max_len:
                k_parts.append(cur_k_cache[..., after_idx:, :, :])
                v_parts.append(cur_v_cache[..., after_idx:, :, :])
            self.cached_key.value = jnp.concatenate(k_parts, axis=-3)
            self.cached_value.value = jnp.concatenate(v_parts, axis=-3)
            self.cache_index.value = after_idx
            k_proj = self.cached_key.value
            v_proj = self.cached_value.value
            if mask is None:
                k_pos = jnp.reshape(jnp.arange(max_len), (1, max_len))
                q_pos = jnp.reshape(jnp.arange(cur_idx, after_idx), (step_len, 1))
                mask = k_pos <= q_pos

        det = bool(deterministic if deterministic is not None else self.deterministic)
        eff_rngs = rngs if rngs is not None else self.rngs
        drop_key = (
            resolve_rng_key(eff_rngs, "dropout") if not det and self.dropout_rate > 0.0 else None
        )

        if self.attention_fn is not None:
            if sow_weights:
                sown_w = _compute_attention_weights(
                    q_proj,
                    k_proj,
                    mask=mask,
                    broadcast_dropout=self.broadcast_dropout,
                    dropout_rng=drop_key,
                    dropout_rate=self.dropout_rate,
                    deterministic=det,
                )
                self.attention_weights = self.register_variable(
                    "attention_weights", Intermediate(sown_w)
                )
            attn_vec: JaxArray = ensure_array(
                self.attention_fn(
                    q_proj,
                    k_proj,
                    v_proj,
                    mask=mask,
                    dropout_rng=drop_key,
                    dropout_rate=self.dropout_rate,
                    broadcast_dropout=self.broadcast_dropout,
                    deterministic=det,
                    dtype=self.dtype,
                    precision=self.precision,
                )
            )
        else:
            attn_vec = dot_product_attention(
                q_proj,
                k_proj,
                v_proj,
                mask=mask,
                dropout_rng=drop_key,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=det,
                dtype=self.dtype,
                precision=self.precision,
                module=self if sow_weights else None,
            )
            if sow_weights:
                sown_var = self._variables["attention_weights"]
                self.attention_weights = self.register_variable(
                    "attention_weights",
                    Intermediate(ensure_array(sown_var.value)),
                )

        output = self.out(attn_vec)
        if added_seq_dim and orig_ndim == 2:
            return output[:, 0, :]
        if added_seq_dim and orig_ndim == 1:
            return output[0, 0, :]
        return output


class MultiHeadDotProductAttention(MultiHeadAttention):
    r"""Multi-Head Dot-Product Attention layer (alias subclass of ``MultiHeadAttention``)."""


class RoPE(Module):
    r"""Rotary Positional Embedding (`RoPE`) applying complex-plane frequency rotations.

    Attributes:
        theta: Base wavelength period $	heta$ for rotary frequencies.
        embedding_size: Optional feature dimension size to rotate.
        max_seq_len: Optional maximum sequence length.
    """

    theta: float
    embedding_size: Optional[int]
    max_seq_len: Optional[int]

    def __init__(
        self,
        theta: float = 10000.0,
        embedding_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        r"""Initialize a RoPE positional embedding module.

        Args:
            theta: Base rotational period $	heta$. Defaults to ``10000.0``.
            embedding_size: Optional even feature dimension size (inferred from input if ``None``).
            max_seq_len: Optional maximum sequence length cap.

        Raises:
            ValueError: If ``embedding_size`` is provided and is not a positive even integer.
        """
        super().__init__()
        self.theta = float(theta)
        if embedding_size is not None:
            emb_sz = int(embedding_size)
            if emb_sz <= 0 or emb_sz % 2 != 0:
                raise ValueError(
                    f"RoPE embedding_size must be a positive even integer, got {embedding_size}."
                )
            self.embedding_size = emb_sz
        else:
            self.embedding_size = None
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.finish_init()

    def __call__(
        self,
        inputs: ArrayLike,
        positions: Optional[ArrayLike] = None,
    ) -> JaxArray:
        r"""Apply Rotary Positional Embedding to ``inputs``.

        Supports 2D ``(seq_len, features)``, 3D ``(batch, seq_len, features)``, and
        4D ``(batch, seq_len, num_heads, head_dim)`` tensors.

        Args:
            inputs: Input tensor of shape ``(*batch, seq_len, features)`` or
                ``(*batch, seq_len, num_heads, head_dim)``.
            positions: Optional explicit position indices of shape ``(seq_len,)`` or ``(*batch, seq_len)``.

        Returns:
            Position-rotated ``JaxArray`` of the same shape and dtype as ``inputs``.

        Raises:
            ValueError: If ``inputs`` has fewer than ``embedding_size`` features in its last dimension.
        """
        x = ensure_array(inputs)
        if x.ndim < 2:
            return x
        if self.embedding_size is not None and x.shape[-1] < self.embedding_size:
            raise ValueError(
                f"RoPE input last dimension ({x.shape[-1]}) is smaller than "
                f"embedding_size ({self.embedding_size})."
            )
        _assert_axis_dim(x, -1, x.shape[-1])
        feat_dim = self.embedding_size if self.embedding_size is not None else x.shape[-1]
        half_dim = feat_dim // 2
        if half_dim == 0:
            return x

        seq_axis = -3 if x.ndim >= 4 else -2
        seq_len = x.shape[seq_axis]

        if positions is not None:
            pos = ensure_array(positions, dtype="float32")
        else:
            pos = jnp.arange(seq_len, dtype="float32")
        if self.max_seq_len is not None:
            pos = jnp.minimum(pos, float(max(0, self.max_seq_len - 1)))

        freq_indices = jnp.arange(half_dim, dtype="float32")
        inv_freq = 1.0 / (self.theta ** (2.0 * freq_indices / float(max(2, 2 * half_dim))))
        angles = jnp.outer(jnp.reshape(pos, (-1,)), inv_freq)
        in_dt = sanitize_dtype_str(str(x.dtype).lower())
        cos_val = ensure_array(jnp.cos(angles), dtype=in_dt)
        sin_val = ensure_array(jnp.sin(angles), dtype=in_dt)

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim : 2 * half_dim]
        bcast_shape = [1] * x1.ndim
        pos_elems = int(math.prod(pos.shape)) if pos.ndim > 0 else 1
        if x.ndim >= 3 and pos.ndim == 2 and pos.shape[0] == x.shape[0] and pos.shape[1] == seq_len:
            bcast_shape[0] = x.shape[0]
            bcast_shape[seq_axis] = seq_len
        elif pos_elems == seq_len:
            bcast_shape[seq_axis] = seq_len
        bcast_shape[-1] = half_dim

        cos_b = jnp.reshape(cos_val, tuple(bcast_shape))
        sin_b = jnp.reshape(sin_val, tuple(bcast_shape))
        rot_x1 = x1 * cos_b - x2 * sin_b
        rot_x2 = x1 * sin_b + x2 * cos_b
        rotated = jnp.concatenate([rot_x1, rot_x2], axis=-1)
        if x.shape[-1] > 2 * half_dim:
            rest = x[..., 2 * half_dim :]
            return jnp.concatenate([rotated, rest], axis=-1)
        return rotated


class PReLU(Module):
    r"""Parametric Rectified Linear Unit ($	ext{PReLU}(x) = \max(0, x) + a \min(0, x)$).

    Attributes:
        negative_slope_init: Initial value for the learnable negative slope.
        dtype: Computation data type.
        param_dtype: Parameter initialization data type.
        negative_slope: Learnable slope ``Param``.
    """

    negative_slope_init: float
    dtype: Optional[DType]
    param_dtype: DType
    promote_dtype: PromoteDtypeFn
    negative_slope: Param[JaxArray]

    def __init__(
        self,
        negative_slope_init: Union[float, ArrayLike] = 0.01,
        *,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        negative_slope_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize a PReLU activation module.

        Args:
            negative_slope_init: Initial scalar or array value for the negative slope parameter. Defaults to ``0.01``.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            promote_dtype: Dtype promotion callable.
            negative_slope_metadata: Metadata mapping attached to ``negative_slope``.
        """
        super().__init__()
        slope_arr = ensure_array(negative_slope_init, dtype=param_dtype)
        self.negative_slope_init = (
            float(negative_slope_init) if isinstance(negative_slope_init, (int, float)) else 0.01
        )
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.promote_dtype = promote_dtype
        meta = dict(negative_slope_metadata) if negative_slope_metadata is not None else {}
        self.negative_slope = self.register_variable(
            "negative_slope",
            Param(slope_arr, **meta),
        )
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        r"""Apply Parametric ReLU activation to ``inputs``.

        Args:
            inputs: Input tensor of any shape.

        Returns:
            Activated ``JaxArray`` where negative elements are scaled by ``self.negative_slope``.
        """
        x = ensure_array(inputs, dtype=self.dtype)
        slope = ensure_array(self.negative_slope.value, dtype=self.dtype)
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(x, slope, dtype=self.dtype)
            x, slope = promoted[0], promoted[1]
        res: JaxArray = jnp.where(x >= 0.0, x, slope * x)
        return res


__all__ = [
    "MultiHeadAttention",
    "MultiHeadDotProductAttention",
    "PReLU",
    "RoPE",
    "dot_product_attention",
    "ones_init",
    "default_bias_init",
]
