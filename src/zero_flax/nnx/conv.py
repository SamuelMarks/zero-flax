"""Convolutional (`Conv`, `ConvTranspose`) and Embedding (`Embed`) layers for zero-flax NNX.

This module implements n-D spatial convolution, transposed convolution, and
token embedding table lookup layers with exact `flax_nnx_v0.12.10` signatures
delegating to `ml_switcheroo_compiler.ops.conv_general_dilated` and `zero_jax`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import ml_switcheroo_compiler.ops as compiler_ops
from ml_switcheroo_compiler.ops.configs import ConvConfig
from zero_jax import numpy as jnp
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

from zero_flax.nnx.module import Module
from zero_flax.nnx.rnglib import Rngs, RngStream, resolve_rng_key
from zero_flax.nnx.variables import Param
from zero_flax.types import (
    EMPTY_METADATA,
    ArrayLike,
    ConvGeneralDilatedT,
    DType,
    InitializerLike,
    JaxArray,
    MetadataMapping,
    PaddingLike,
    PaddingSpec,
    PrecisionSpec,
    PromoteDtypeFn,
    default_bias_init,
    default_embed_init,
    default_kernel_init,
    default_promote_dtype,
    ensure_array,
    is_padding_pair_sequence,
    normalize_int_tuple,
    resolve_initializer,
    sanitize_dtype_str,
)


def _assert_conv_rank_and_channels(
    x: JaxArray,
    expected_channels: int,
    kernel: JaxArray,
    num_spatial: int,
) -> None:
    """Validate spatial rank, kernel rank, and trailing channel dimension of ``x`` via ``zero_chex``.

    Args:
        x: Input ``JaxArray`` tensor.
        expected_channels: Expected size of the last channel axis.
        kernel: Convolution kernel ``JaxArray``.
        num_spatial: Number of spatial filter dimensions.
    """
    from zero_chex.assertions import assert_axis_dimension, assert_rank

    assert_rank(x, x.ndim)
    assert_axis_dimension(x, -1, expected_channels)
    assert_rank(kernel, num_spatial + 2)


def _compute_spatial_padding(
    in_size: int,
    eff_k: int,
    stride: int,
    pad_val: Union[int, str],
) -> Tuple[int, int]:
    """Compute ``(pad_before, pad_after)`` for a single forward-convolution spatial dimension.

    Args:
        in_size: Input spatial dimension size.
        eff_k: Effective dilated kernel size along this spatial dimension.
        stride: Convolution stride along this spatial dimension.
        pad_val: Padding mode string or explicit integer padding.

    Returns:
        Pair ``(pad_before, pad_after)`` of non-negative integers.

    Raises:
        ValueError: If ``pad_val`` is an unrecognized string mode.
    """
    if isinstance(pad_val, int):
        return (int(pad_val), int(pad_val))
    mode = pad_val.upper()
    if mode == "VALID":
        return (0, 0)
    if mode == "CAUSAL":
        return (eff_k - 1, 0)
    if mode in ("SAME", "CIRCULAR"):
        out_size = (in_size + stride - 1) // stride
        pad_total = max(0, (out_size - 1) * stride + eff_k - in_size)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return (pad_before, pad_after)
    raise ValueError(f"Unsupported convolution padding mode: {pad_val!r}")


def _compute_transpose_spatial_padding(
    eff_k: int,
    stride: int,
    pad_val: Union[int, str],
) -> Tuple[int, int]:
    """Compute ``(pad_before, pad_after)`` for a single transposed-convolution spatial dimension.

    Follows ``jax.lax._conv_transpose_padding`` exact equations.

    Args:
        eff_k: Effective dilated kernel size along this spatial dimension.
        stride: Transposed convolution stride along this spatial dimension.
        pad_val: Padding mode (``"SAME"``, ``"VALID"``, ``"CIRCULAR"``) or integer.

    Returns:
        Pair ``(pad_before, pad_after)`` for the dilated input in transposed convolution.

    Raises:
        ValueError: If ``pad_val`` is an unrecognized string mode.
    """
    if isinstance(pad_val, int):
        return (int(pad_val), int(pad_val))
    mode = pad_val.upper()
    if mode in ("SAME", "CIRCULAR"):
        pad_len = eff_k + stride - 2
        pad_a = eff_k - 1 if stride > eff_k - 1 else (pad_len + 1) // 2
        pad_b = pad_len - pad_a
        return (max(0, pad_a), max(0, pad_b))
    if mode == "VALID":
        pad_len = eff_k + stride - 2 + max(eff_k - stride, 0)
        pad_a = eff_k - 1
        pad_b = pad_len - pad_a
        return (max(0, pad_a), max(0, pad_b))
    raise ValueError(f"Unsupported transposed convolution padding mode: {pad_val!r}")


def _resolve_spatial_pads(
    spatial_shape: Sequence[int],
    kernel_size: Sequence[int],
    kernel_dilation: Sequence[int],
    strides: Sequence[int],
    padding: PaddingSpec,
    is_transpose: bool = False,
) -> Tuple[list[Tuple[int, int]], bool]:
    """Resolve padding specification into per-axis ``(pad_before, pad_after)`` pairs and circular flag.

    Args:
        spatial_shape: Input spatial dimensions.
        kernel_size: Filter spatial dimensions.
        kernel_dilation: Filter dilation factors.
        strides: Convolution strides.
        padding: Padding specification (string mode, int, sequence of ints, or sequence of pairs).
        is_transpose: Whether resolving for ``ConvTranspose`` rather than ``Conv``.

    Returns:
        Pair ``(spatial_pads, is_circular)``.
    """
    pad_val = padding.padding if isinstance(padding, PaddingLike) else padding
    is_circular = isinstance(pad_val, str) and pad_val.upper() == "CIRCULAR"
    num_spatial = len(spatial_shape)

    if is_padding_pair_sequence(pad_val):
        pairs: list[Tuple[int, int]] = []
        for idx in range(num_spatial):
            p_pair = pad_val[idx] if idx < len(pad_val) else pad_val[-1]
            pairs.append((int(p_pair[0]), int(p_pair[1])))
        return pairs, False

    if isinstance(pad_val, Sequence) and not isinstance(pad_val, str):
        seq_pads: list[Tuple[int, int]] = []
        for idx in range(num_spatial):
            raw_el = pad_val[idx] if idx < len(pad_val) else pad_val[-1]
            p_int = int(raw_el[0]) if isinstance(raw_el, tuple) else int(raw_el)
            seq_pads.append((p_int, p_int))
        return seq_pads, False

    computed: list[Tuple[int, int]] = []
    for idx in range(num_spatial):
        eff_k = (kernel_size[idx] - 1) * kernel_dilation[idx] + 1
        if is_transpose:
            computed.append(
                _compute_transpose_spatial_padding(
                    eff_k,
                    strides[idx],
                    pad_val,
                )
            )
        else:
            computed.append(
                _compute_spatial_padding(
                    spatial_shape[idx],
                    eff_k,
                    strides[idx],
                    pad_val,
                )
            )
    return computed, is_circular


def _pad_circular_spatial_nd(
    x: JaxArray,
    spatial_pads: Sequence[Tuple[int, int]],
) -> JaxArray:
    """Wrap spatial axes of ``x`` modularly from opposite boundaries according to ``spatial_pads``.

    Args:
        x: Input array of shape ``(batch, *spatial, channels)``.
        spatial_pads: Sequence of ``(pad_before, pad_after)`` pairs for each spatial axis.

    Returns:
        Circularly padded ``JaxArray``.
    """
    cur = x
    for idx, (pad_before, pad_after) in enumerate(spatial_pads):
        axis = 1 + idx
        orig_axis_len = cur.shape[axis]
        if pad_before > 0 or pad_after > 0:
            pieces: list[JaxArray] = []
            for p_idx in range(-pad_before, orig_axis_len + pad_after):
                wrap_idx = p_idx % orig_axis_len
                slices = [slice(None)] * cur.ndim
                slices[axis] = slice(wrap_idx, wrap_idx + 1)
                pieces.append(cur[tuple(slices)])
            cur = jnp.concatenate(pieces, axis=axis)
    return cur


def _execute_compiler_conv_general_dilated(
    lhs_nhwc: JaxArray,
    rhs_hwio: JaxArray,
    window_strides: Tuple[int, ...],
    padding_pairs: list[Tuple[int, int]],
    lhs_dilation: Tuple[int, ...],
    rhs_dilation: Tuple[int, ...],
    feature_group_count: int = 1,
) -> JaxArray:
    """Execute n-D dilated convolution via ``ml_switcheroo_compiler.ops.conv_general_dilated``.

    Args:
        lhs_nhwc: Input tensor in ``(N, *spatial, C_in)`` layout.
        rhs_hwio: Filter tensor in ``(*spatial, C_in_per_group, C_out)`` layout.
        window_strides: Spatial convolution strides.
        padding_pairs: Per-axis ``(pad_before, pad_after)`` padding pairs.
        lhs_dilation: Spatial input dilation factors.
        rhs_dilation: Spatial filter dilation factors.
        feature_group_count: Number of channel feature groups.

    Returns:
        Output ``JaxArray`` in ``(N, *out_spatial, C_out)`` layout.
    """
    num_spatial = len(window_strides)
    lhs_perm = (0, num_spatial + 1) + tuple(range(1, num_spatial + 1))
    rhs_perm = (num_spatial + 1, num_spatial) + tuple(range(num_spatial))
    cfg = ConvConfig(
        window_strides=window_strides,
        padding=padding_pairs,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=(lhs_perm, rhs_perm, lhs_perm),
        feature_group_count=feature_group_count,
    )
    out_tensor = compiler_ops.conv_general_dilated(
        _to_tensor(lhs_nhwc),
        _to_tensor(rhs_hwio),
        cfg,
    )
    res: JaxArray = _wrap(out_tensor)
    return res


class Conv(Module):
    """General n-D spatial convolution layer.

    Attributes:
        in_features: Number of input channels.
        out_features: Number of output channels.
        kernel_size: Spatial dimensions of the convolutional filter.
        strides: Spatial strides of the convolution.
        padding: Spatial padding mode or explicit padding specification.
        input_dilation: Input dilation factors.
        kernel_dilation: Kernel dilation (atrous rate) factors.
        feature_group_count: Number of channel groups.
        use_bias: Whether a learnable bias vector is added.
        kernel: Learnable convolution filter ``Param``.
        bias: Optional learnable bias ``Param``.
    """

    in_features: int
    out_features: int
    kernel_size: Tuple[int, ...]
    strides: Tuple[int, ...]
    padding: PaddingSpec
    input_dilation: Tuple[int, ...]
    kernel_dilation: Tuple[int, ...]
    feature_group_count: int
    use_bias: bool
    mask: Optional[JaxArray]
    dtype: Optional[DType]
    param_dtype: DType
    conv_general_dilated: ConvGeneralDilatedT
    precision: PrecisionSpec
    promote_dtype: PromoteDtypeFn
    preferred_element_type: Optional[DType]
    kernel: Param[JaxArray]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[None, int, Sequence[int]] = 1,
        *,
        padding: PaddingSpec = "SAME",
        input_dilation: Union[None, int, Sequence[int]] = None,
        kernel_dilation: Union[None, int, Sequence[int]] = None,
        feature_group_count: Optional[int] = 1,
        use_bias: bool = True,
        mask: Optional[ArrayLike] = None,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        conv_general_dilated: ConvGeneralDilatedT = None,
        precision: PrecisionSpec = None,
        kernel_init: InitializerLike = default_kernel_init,
        bias_init: InitializerLike = default_bias_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        preferred_element_type: Optional[DType] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize a Conv layer.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels.
            kernel_size: Integer or sequence of integers specifying spatial filter dimensions.
            strides: Integer or sequence of integers specifying spatial strides.
            padding: Padding mode (``"SAME"``, ``"VALID"``, ``"CAUSAL"``, ``"CIRCULAR"``)
                or explicit padding tuples.
            input_dilation: Optional spatial input dilation factors.
            kernel_dilation: Optional spatial filter dilation factors.
            feature_group_count: Number of groups for grouped convolution.
            use_bias: Whether to add a learnable bias vector.
            mask: Optional multiplicative mask applied to the convolution kernel.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            conv_general_dilated: Optional custom dilated convolution callable.
            precision: Numerical precision specification.
            kernel_init: Filter weight initializer.
            bias_init: Bias vector initializer.
            promote_dtype: Dtype promotion callable.
            preferred_element_type: Optional accumulation dtype.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata attached to ``kernel``.
            bias_metadata: Metadata attached to ``bias``.

        Raises:
            ValueError: If ``in_features`` or ``out_features`` is not divisible by ``feature_group_count``.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size = (
            (int(kernel_size),)
            if isinstance(kernel_size, int)
            else tuple(int(k) for k in kernel_size)
        )
        num_spatial = len(self.kernel_size)
        self.strides = normalize_int_tuple(strides, num_spatial, default=1)
        self.padding = padding
        self.input_dilation = normalize_int_tuple(input_dilation, num_spatial, default=1)
        self.kernel_dilation = normalize_int_tuple(kernel_dilation, num_spatial, default=1)
        self.feature_group_count = (
            int(feature_group_count) if feature_group_count is not None else 1
        )
        if self.feature_group_count <= 0:
            raise ValueError("feature_group_count must be a positive integer.")
        if self.in_features % self.feature_group_count != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by "
                f"feature_group_count ({self.feature_group_count})."
            )
        if self.out_features % self.feature_group_count != 0:
            raise ValueError(
                f"out_features ({self.out_features}) must be divisible by "
                f"feature_group_count ({self.feature_group_count})."
            )
        self.use_bias = bool(use_bias)
        self.mask = None if mask is None else ensure_array(mask)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.conv_general_dilated = conv_general_dilated
        self.precision = precision
        self.promote_dtype = promote_dtype
        self.preferred_element_type = preferred_element_type

        k_fn = resolve_initializer(kernel_init, default_kernel_init)
        k_key = resolve_rng_key(rngs, "params", fold_index=0)
        k_meta = dict(kernel_metadata) if kernel_metadata is not None else {}
        k_shape = self.kernel_size + (
            self.in_features // self.feature_group_count,
            self.out_features,
        )
        self.kernel = self.register_variable(
            "kernel",
            Param(k_fn(k_key, k_shape, param_dtype), **k_meta),
        )

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, default_bias_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, (self.out_features,), param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        """Apply n-D spatial convolution to ``inputs``.

        Args:
            inputs: Input tensor of shape ``(*batch_dims, *spatial_dims, in_features)``
                or ``(*spatial_dims, in_features)``.

        Returns:
            Convolved ``JaxArray`` with ``out_features`` channels in the last dimension.

        Raises:
            ValueError: If ``inputs`` has fewer than ``len(kernel_size) + 1`` dimensions,
                its channel dimension does not match ``self.in_features``, or the padded
                spatial dimension is smaller than the effective kernel size.
        """
        x = ensure_array(inputs, dtype=self.dtype)
        k = ensure_array(self.kernel.value, dtype=self.dtype)
        if self.mask is not None:
            k = k * self.mask
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype)
            if (self.use_bias and self.bias is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(x, k, b_raw, dtype=self.dtype)
            x, k = promoted[0], promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]

        num_spatial = len(self.kernel_size)
        if x.ndim < num_spatial + 1:
            raise ValueError(
                f"Conv expects input with at least {num_spatial + 1} dimensions "
                f"(*spatial, in_features), got shape {x.shape}."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Conv input channel dimension ({x.shape[-1]}) does not match "
                f"in_features ({self.in_features})."
            )
        _assert_conv_rank_and_channels(x, self.in_features, k, num_spatial)

        if self.conv_general_dilated is not None:
            custom_out = ensure_array(self.conv_general_dilated(x, k))
            if b_raw is not None:
                custom_out = custom_out + b_raw
            return custom_out

        has_batch = x.ndim > num_spatial + 1
        batch_shape = x.shape[: x.ndim - num_spatial - 1] if has_batch else ()
        spatial_shape = x.shape[x.ndim - num_spatial - 1 : -1]
        in_ch = x.shape[-1]
        batch_size = 1
        for b_dim in batch_shape:
            batch_size *= b_dim
        x_batched = jnp.reshape(x, (batch_size, *spatial_shape, in_ch))

        dilated_spatial = tuple(
            max(1, (spatial_shape[i] - 1) * self.input_dilation[i] + 1) for i in range(num_spatial)
        )
        spatial_pads, is_circular = _resolve_spatial_pads(
            dilated_spatial,
            self.kernel_size,
            self.kernel_dilation,
            self.strides,
            self.padding,
            is_transpose=False,
        )
        for idx in range(num_spatial):
            eff_k = (self.kernel_size[idx] - 1) * self.kernel_dilation[idx] + 1
            padded_size = dilated_spatial[idx] + spatial_pads[idx][0] + spatial_pads[idx][1]
            if padded_size < eff_k:
                raise ValueError(
                    f"Conv padded spatial size ({padded_size}) is smaller than "
                    f"effective kernel size ({eff_k}) along spatial axis {idx}."
                )

        if is_circular:
            if any(d > 1 for d in self.input_dilation):
                x_batched = _execute_compiler_conv_general_dilated(
                    x_batched,
                    jnp.eye(in_ch, dtype=sanitize_dtype_str(str(x.dtype).lower())).reshape(
                        (1,) * num_spatial + (in_ch, in_ch)
                    ),
                    window_strides=(1,) * num_spatial,
                    padding_pairs=[(0, 0)] * num_spatial,
                    lhs_dilation=self.input_dilation,
                    rhs_dilation=(1,) * num_spatial,
                    feature_group_count=1,
                )
            x_batched = _pad_circular_spatial_nd(x_batched, spatial_pads)
            eff_input_dilation = (1,) * num_spatial
            eff_pads = [(0, 0)] * num_spatial
        else:
            eff_input_dilation = self.input_dilation
            eff_pads = spatial_pads

        acc = _execute_compiler_conv_general_dilated(
            x_batched,
            k,
            window_strides=self.strides,
            padding_pairs=eff_pads,
            lhs_dilation=eff_input_dilation,
            rhs_dilation=self.kernel_dilation,
            feature_group_count=self.feature_group_count,
        )

        if b_raw is not None:
            acc = acc + b_raw

        if self.preferred_element_type is not None:
            acc = ensure_array(acc, dtype=self.preferred_element_type)

        out_spatial = acc.shape[1:-1]
        final_shape = (*batch_shape, *out_spatial, self.out_features)
        return jnp.reshape(acc, final_shape)


class ConvTranspose(Module):
    """General n-D transposed spatial convolution (deconvolution) layer.

    Attributes:
        in_features: Number of input channels.
        out_features: Number of output channels.
        kernel_size: Tuple of spatial filter dimensions.
        strides: Tuple of spatial strides.
        padding: Padding mode or explicit padding tuples.
        kernel_dilation: Tuple of spatial kernel dilation factors.
        use_bias: Whether a learnable bias vector is added.
        transpose_kernel: Whether to transpose the spatial kernel dimensions.
        kernel: Learnable filter ``Param``.
        bias: Optional learnable bias ``Param``.
    """

    in_features: int
    out_features: int
    kernel_size: Tuple[int, ...]
    strides: Tuple[int, ...]
    padding: PaddingSpec
    kernel_dilation: Tuple[int, ...]
    use_bias: bool
    mask: Optional[JaxArray]
    dtype: Optional[DType]
    param_dtype: DType
    precision: PrecisionSpec
    transpose_kernel: bool
    promote_dtype: PromoteDtypeFn
    preferred_element_type: Optional[DType]
    kernel: Param[JaxArray]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int], None] = 1,
        *,
        padding: PaddingSpec = "SAME",
        kernel_dilation: Union[int, Sequence[int], None] = None,
        bias_init: InitializerLike = default_bias_init,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        use_bias: bool = True,
        dtype: Optional[DType] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        param_dtype: DType = "float32",
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        kernel_init: InitializerLike = default_kernel_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        preferred_element_type: Optional[DType] = None,
        mask: Optional[ArrayLike] = None,
        precision: PrecisionSpec = None,
        transpose_kernel: bool = False,
    ) -> None:
        """Initialize a ConvTranspose layer.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels.
            kernel_size: Spatial filter size or sequence of sizes.
            strides: Spatial stride or sequence of strides.
            padding: Spatial padding specification (``"SAME"``, ``"VALID"``, ``"CIRCULAR"``).
            kernel_dilation: Optional spatial filter dilation factors.
            bias_init: Initializer for the bias vector.
            rngs: Random number generator stream(s).
            use_bias: Whether to add a learnable bias vector.
            dtype: Computation data type.
            kernel_metadata: Metadata mapping for ``kernel``.
            param_dtype: Parameter initialization data type.
            bias_metadata: Metadata mapping for ``bias``.
            kernel_init: Initializer for the filter kernel.
            promote_dtype: Dtype promotion callable.
            preferred_element_type: Optional accumulation dtype.
            mask: Optional multiplicative kernel mask.
            precision: Numerical precision specification.
            transpose_kernel: Whether to flip/transpose kernel spatial axes.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size = (
            (int(kernel_size),)
            if isinstance(kernel_size, int)
            else tuple(int(k) for k in kernel_size)
        )
        num_spatial = len(self.kernel_size)
        self.strides = normalize_int_tuple(strides, num_spatial, default=1)
        self.padding = padding
        self.kernel_dilation = normalize_int_tuple(kernel_dilation, num_spatial, default=1)
        self.use_bias = bool(use_bias)
        self.mask = None if mask is None else ensure_array(mask)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.transpose_kernel = bool(transpose_kernel)
        self.promote_dtype = promote_dtype
        self.preferred_element_type = preferred_element_type

        k_fn = resolve_initializer(kernel_init, default_kernel_init)
        k_key = resolve_rng_key(rngs, "params", fold_index=0)
        k_meta = dict(kernel_metadata) if kernel_metadata is not None else {}
        if self.transpose_kernel:
            k_shape = self.kernel_size + (self.out_features, self.in_features)
        else:
            k_shape = self.kernel_size + (self.in_features, self.out_features)
        self.kernel = self.register_variable(
            "kernel",
            Param(k_fn(k_key, k_shape, param_dtype), **k_meta),
        )

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, default_bias_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, (self.out_features,), param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        """Apply n-D transposed spatial convolution to ``inputs``.

        Implements fractionally strided convolution matching ``jax.lax.conv_transpose``
        by delegating to ``ml_switcheroo_compiler.ops.conv_general_dilated`` with
        input dilation ``lhs_dilation = self.strides``.

        Args:
            inputs: Input tensor of shape ``(*batch, *spatial, in_features)``
                or ``(*spatial, in_features)``.

        Returns:
            Transposed-convolved ``JaxArray`` with ``out_features`` channels.

        Raises:
            ValueError: If ``inputs`` has fewer than ``len(kernel_size) + 1`` dimensions or
                its channel dimension does not match ``self.in_features``.
        """
        x = ensure_array(inputs, dtype=self.dtype)
        k = ensure_array(self.kernel.value, dtype=self.dtype)
        if self.mask is not None:
            k = k * self.mask
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype)
            if (self.use_bias and self.bias is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(x, k, b_raw, dtype=self.dtype)
            x, k = promoted[0], promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]

        num_spatial = len(self.kernel_size)
        if x.ndim < num_spatial + 1:
            raise ValueError(
                f"ConvTranspose expects input with at least {num_spatial + 1} dimensions "
                f"(*spatial, in_features), got shape {x.shape}."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"ConvTranspose input channel dimension ({x.shape[-1]}) does not match "
                f"in_features ({self.in_features})."
            )
        _assert_conv_rank_and_channels(x, self.in_features, k, num_spatial)

        if self.transpose_kernel:
            flip_slices = tuple(slice(None, None, -1) for _ in range(num_spatial)) + (
                slice(None),
                slice(None),
            )
            k_flipped = k[flip_slices]
            perm = tuple(range(num_spatial)) + (num_spatial + 1, num_spatial)
            k_eff = jnp.transpose(k_flipped, perm)
        else:
            k_eff = k

        has_batch = x.ndim > num_spatial + 1
        batch_shape = x.shape[: x.ndim - num_spatial - 1] if has_batch else ()
        spatial_shape = x.shape[x.ndim - num_spatial - 1 : -1]
        in_ch = x.shape[-1]
        batch_size = 1
        for b_dim in batch_shape:
            batch_size *= b_dim
        x_batched = jnp.reshape(x, (batch_size, *spatial_shape, in_ch))

        spatial_pads, is_circular = _resolve_spatial_pads(
            spatial_shape,
            self.kernel_size,
            self.kernel_dilation,
            self.strides,
            self.padding,
            is_transpose=True,
        )

        if is_circular:
            if any(s > 1 for s in self.strides):
                x_batched = _execute_compiler_conv_general_dilated(
                    x_batched,
                    jnp.eye(in_ch, dtype=sanitize_dtype_str(str(x.dtype).lower())).reshape(
                        (1,) * num_spatial + (in_ch, in_ch)
                    ),
                    window_strides=(1,) * num_spatial,
                    padding_pairs=[(0, 0)] * num_spatial,
                    lhs_dilation=self.strides,
                    rhs_dilation=(1,) * num_spatial,
                    feature_group_count=1,
                )
            x_batched = _pad_circular_spatial_nd(x_batched, spatial_pads)
            eff_strides = (1,) * num_spatial
            eff_pads = [(0, 0)] * num_spatial
        else:
            eff_strides = self.strides
            eff_pads = spatial_pads

        acc = _execute_compiler_conv_general_dilated(
            x_batched,
            k_eff,
            window_strides=(1,) * num_spatial,
            padding_pairs=eff_pads,
            lhs_dilation=eff_strides,
            rhs_dilation=self.kernel_dilation,
            feature_group_count=1,
        )

        if b_raw is not None:
            acc = acc + b_raw

        if self.preferred_element_type is not None:
            acc = ensure_array(acc, dtype=self.preferred_element_type)

        out_spatial = acc.shape[1:-1]
        return jnp.reshape(acc, (*batch_shape, *out_spatial, self.out_features))


class Embed(Module):
    """Parameterized token embedding lookup table layer.

    Attributes:
        num_embeddings: Vocabulary size (number of rows in the embedding matrix).
        features: Embedding vector dimension size.
        dtype: Output embedding data type.
        param_dtype: Parameter initialization data type.
        embedding: Learnable embedding matrix ``Param`` of shape ``(num_embeddings, features)``.
    """

    num_embeddings: int
    features: int
    dtype: Optional[DType]
    param_dtype: DType
    embedding_init: InitializerLike
    promote_dtype: PromoteDtypeFn
    embedding: Param[JaxArray]

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        embedding_init: InitializerLike = default_embed_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        embedding_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize an Embed lookup layer.

        Args:
            num_embeddings: Number of distinct embeddings in the vocabulary.
            features: Dimensionality of each embedding vector.
            dtype: Output data type of looked-up embeddings.
            param_dtype: Data type of the embedding parameter table.
            embedding_init: Initializer function for the embedding table.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            embedding_metadata: Optional metadata mapping attached to ``embedding``.
        """
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.features = int(features)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.embedding_init = embedding_init
        self.promote_dtype = promote_dtype

        e_fn = resolve_initializer(embedding_init, default_embed_init)
        e_key = resolve_rng_key(rngs, "params", fold_index=0)
        e_meta = dict(embedding_metadata) if embedding_metadata is not None else {}
        self.embedding = self.register_variable(
            "embedding",
            Param(e_fn(e_key, (self.num_embeddings, self.features), param_dtype), **e_meta),
        )
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        """Look up embeddings for integer token indices ``inputs``.

        Args:
            inputs: Integer array of token IDs in ``[0, num_embeddings)``.

        Returns:
            Embedded ``JaxArray`` of shape ``(*inputs.shape, features)``.
        """
        from zero_chex.assertions import assert_type

        idx = ensure_array(inputs, dtype="int32")
        assert_type(idx, int)
        table = ensure_array(self.embedding.value, dtype=self.dtype)
        if self.promote_dtype is not None:
            table = self.promote_dtype(table, dtype=self.dtype)[0]
        res: JaxArray = table[idx]
        return res

    def attend(self, query: ArrayLike) -> JaxArray:
        """Compute logit similarity scores between ``query`` and the embedding table ($Q E^T$).

        Args:
            query: Query activation array with last dimension ``features``.

        Returns:
            Logit ``JaxArray`` with last dimension ``num_embeddings``.
        """
        from zero_chex.assertions import assert_axis_dimension

        q = ensure_array(query)
        assert_axis_dimension(q, -1, self.features)
        table = ensure_array(self.embedding.value)
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(q, table, dtype=self.dtype)
            q, table = promoted[0], promoted[1]
        res: JaxArray = jnp.dot(q, jnp.transpose(table))
        return res


__all__ = [
    "Conv",
    "ConvTranspose",
    "Embed",
]
