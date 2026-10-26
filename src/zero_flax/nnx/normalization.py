"""Normalization layers and weight wrappers for the zero-flax NNX API.

This module implements `BatchNorm`, `LayerNorm`, `RMSNorm`, `GroupNorm`,
`InstanceNorm`, `SpectralNorm`, and `WeightNorm` with exact `flax_nnx_v0.12.10`
parameter signatures and mathematical forward-pass execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, Union

from zero_jax import lax
from zero_jax import numpy as jnp
from zero_jax import random as jrandom

from zero_flax.nnx.filterlib import FilterLiteral, PathContains, to_predicate
from zero_flax.nnx.module import Module, iter_graph
from zero_flax.nnx.rnglib import Rngs, RngStream, resolve_rng_key
from zero_flax.nnx.variables import AnyVariable, BatchStat, Param
from zero_flax.types import (
    EMPTY_METADATA,
    ArrayLike,
    Axes,
    AxisIndexGroups,
    DType,
    InitializerLike,
    JaxArray,
    MetadataMapping,
    PromoteDtypeFn,
    default_promote_dtype,
    ensure_array,
    ones_init,
    resolve_initializer,
    zeros_init,
)


class _UnsetGroupSentinel:
    """Sentinel class distinguishing omitted ``num_groups`` from an explicit integer."""


_UNSET_GROUPS = _UnsetGroupSentinel()


def _invoke_wrapped_layer(layer: Module, x: ArrayLike) -> JaxArray:
    """Invoke a callable ``Module`` instance on ``x`` and ensure a ``JaxArray`` result.

    Args:
        layer: Wrapped ``Module`` instance to evaluate.
        x: Input array operand.

    Returns:
        Evaluated output ``JaxArray``.
    """
    res: ArrayLike = layer(x) if callable(layer) else x
    return ensure_array(res)


def _normalize_axes(axes: Union[Axes, int, Sequence[int], None], ndim: int) -> Tuple[int, ...]:
    r"""Normalize an ``Axes``, integer, or sequence of integers into positive axis indices.

    Args:
        axes: Axis specification to normalize.
        ndim: Number of dimensions of the target tensor.

    Returns:
        Tuple of non-negative axis indices.
    """
    if axes is None:
        return (max(0, ndim - 1),)
    if isinstance(axes, Axes):
        raw = axes.axes
    elif isinstance(axes, int):
        raw = (axes,)
    else:
        raw = tuple(int(a) for a in axes)
    return tuple(a if a >= 0 else max(0, ndim + a) for a in raw)


def _assert_feature_dim(tensor: JaxArray, axis: int, expected: int) -> None:
    """Validate that ``tensor`` has size ``expected`` along ``axis`` via ``zero_chex``.

    Args:
        tensor: Input ``JaxArray`` to validate.
        axis: Axis index to check.
        expected: Expected dimension size along ``axis``.
    """
    from zero_chex.assertions import assert_axis_dimension

    assert_axis_dimension(tensor, axis, expected)


def _aggregate_moments_across_devices(
    mean: JaxArray,
    var: JaxArray,
    axis_name: Optional[str],
    axis_index_groups: AxisIndexGroups,
) -> Tuple[JaxArray, JaxArray]:
    """Aggregate ``(mean, var)`` across parallel devices via ``zero_jax.lax.pmean``.

    Args:
        mean: Local mean tensor.
        var: Local variance tensor.
        axis_name: Optional parallel axis name.
        axis_index_groups: Optional groups of device axis indices.

    Returns:
        Aggregated ``(mean, var)`` pair.
    """
    if axis_name is None:
        return mean, var
    agg_mean: JaxArray = ensure_array(
        lax.pmean(mean, axis_name=axis_name, axis_index_groups=axis_index_groups)
    )
    agg_var: JaxArray = ensure_array(
        lax.pmean(var, axis_name=axis_name, axis_index_groups=axis_index_groups)
    )
    return agg_mean, agg_var


def _compute_moments(
    arr: JaxArray,
    red_axes: Tuple[int, ...],
    mask: Optional[ArrayLike] = None,
    keepdims: bool = True,
    subtract_mean: bool = True,
    use_fast_variance: bool = True,
    axis_name: Optional[str] = None,
    axis_index_groups: AxisIndexGroups = None,
) -> Tuple[JaxArray, JaxArray]:
    r"""Compute (optionally masked) mean and variance of ``arr`` along ``red_axes``.

    Args:
        arr: Input tensor.
        red_axes: Tuple of non-negative reduction axis indices.
        mask: Optional boolean/numeric mask broadcastable to ``arr.shape``.
        keepdims: Whether reduced axes are retained with size ``1``.
        subtract_mean: Whether to center ``arr`` by its mean when computing variance.
        use_fast_variance: If ``True`` and ``subtract_mean`` is ``True``, compute variance via
            $\max(0, \mathbb{E}[x^2] - \mathbb{E}[x]^2)$; otherwise compute $\mathbb{E}[(x - \mu)^2]$.
        axis_name: Optional parallel axis name for ``lax.pmean`` aggregation.
        axis_index_groups: Optional device index groups for ``lax.pmean``.

    Returns:
        Pair ``(mean, var)`` of ``JaxArray`` tensors.
    """
    if len(red_axes) == 0:
        return arr, jnp.zeros(arr.shape)
    if mask is None:
        mean: JaxArray = jnp.mean(arr, axis=red_axes, keepdims=keepdims)
        if subtract_mean:
            if use_fast_variance:
                mean_sq: JaxArray = jnp.mean(arr * arr, axis=red_axes, keepdims=keepdims)
                var: JaxArray = jnp.maximum(0.0, mean_sq - mean * mean)
            else:
                mean_kd = mean if keepdims else jnp.expand_dims(mean, axis=red_axes)
                diff = arr - mean_kd
                var = jnp.mean(diff * diff, axis=red_axes, keepdims=keepdims)
        else:
            var = jnp.mean(arr * arr, axis=red_axes, keepdims=keepdims)
        return _aggregate_moments_across_devices(mean, var, axis_name, axis_index_groups)

    m_arr = ensure_array(mask, dtype="float32")
    m_b: JaxArray = jnp.zeros(arr.shape, dtype="float32") + m_arr
    denom: JaxArray = jnp.maximum(jnp.sum(m_b, axis=red_axes, keepdims=keepdims), 1.0)
    mean = jnp.sum(arr * m_b, axis=red_axes, keepdims=keepdims) / denom
    if subtract_mean:
        if use_fast_variance:
            mean_sq = jnp.sum(arr * arr * m_b, axis=red_axes, keepdims=keepdims) / denom
            var = jnp.maximum(0.0, mean_sq - mean * mean)
        else:
            mean_kd = mean if keepdims else jnp.expand_dims(mean, axis=red_axes)
            diff = (arr - mean_kd) * m_b
            var = jnp.sum(diff * diff, axis=red_axes, keepdims=keepdims) / denom
    else:
        var = jnp.sum(arr * arr * m_b, axis=red_axes, keepdims=keepdims) / denom
    return _aggregate_moments_across_devices(mean, var, axis_name, axis_index_groups)


class BatchNorm(Module):
    r"""Batch Normalization layer normalizing activations across batch and spatial axes.

    Attributes:
        num_features: Number of feature channels.
        use_running_average: Default flag controlling whether stored running statistics are used.
        axis: Feature/channel axis index.
        momentum: Exponential moving average decay factor for running statistics.
        epsilon: Small constant added to variance for numerical stability.
        use_bias: Whether a learnable shift parameter is applied.
        use_scale: Whether a learnable scale parameter is applied.
        mean: Running mean ``BatchStat`` of shape ``(num_features,)``.
        var: Running variance ``BatchStat`` of shape ``(num_features,)``.
        scale: Optional learnable scale ``Param`` of shape ``(num_features,)``.
        bias: Optional learnable bias ``Param`` of shape ``(num_features,)``.
    """

    num_features: int
    use_running_average: Optional[bool]
    axis: int
    momentum: float
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    use_bias: bool
    use_scale: bool
    subtract_mean: bool
    axis_name: Optional[str]
    axis_index_groups: AxisIndexGroups
    use_fast_variance: Optional[bool]
    promote_dtype: PromoteDtypeFn
    mean: BatchStat[JaxArray]
    var: BatchStat[JaxArray]
    scale: Optional[Param[JaxArray]]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        num_features: int,
        *,
        use_running_average: Optional[bool] = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-05,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        use_bias: bool = True,
        use_scale: bool = True,
        subtract_mean: bool = True,
        bias_init: InitializerLike = zeros_init,
        scale_init: InitializerLike = ones_init,
        axis_name: Optional[str] = None,
        axis_index_groups: AxisIndexGroups = None,
        use_fast_variance: Optional[bool] = True,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        scale_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize a BatchNorm layer.

        Args:
            num_features: Size of the feature dimension.
            use_running_average: If ``True``, use stored ``mean`` and ``var`` statistics;
                if ``False``, compute batch statistics and update ``mean`` and ``var``.
            axis: Index of the feature/channel axis. Defaults to ``-1``.
            momentum: Decay rate for the exponential moving average of batch statistics.
            epsilon: Small float added to variance to avoid division by zero.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            use_bias: Whether to include a learnable additive bias parameter.
            use_scale: Whether to include a learnable multiplicative scale parameter.
            subtract_mean: Whether to center activations by subtracting the mean.
            bias_init: Initializer for the bias parameter.
            scale_init: Initializer for the scale parameter.
            axis_name: Optional parallel axis name for cross-device aggregation.
            axis_index_groups: Optional groups of axis indices for distributed aggregation.
            use_fast_variance: Whether to compute variance as $\mathbb{E}[x^2] - \mathbb{E}[x]^2$.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            bias_metadata: Metadata mapping attached to ``bias``.
            scale_metadata: Metadata attached to ``scale``.
        """
        super().__init__()
        self.num_features = int(num_features)
        self.use_running_average = use_running_average
        self.axis = int(axis)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = bool(use_bias)
        self.use_scale = bool(use_scale)
        self.subtract_mean = bool(subtract_mean)
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance
        self.promote_dtype = promote_dtype

        self.mean = self.register_variable(
            "mean",
            BatchStat(ensure_array(jnp.zeros((self.num_features,)), dtype=param_dtype)),
        )
        self.var = self.register_variable(
            "var",
            BatchStat(ensure_array(jnp.ones((self.num_features,)), dtype=param_dtype)),
        )

        if self.use_scale:
            s_fn = resolve_initializer(scale_init, ones_init)
            s_key = resolve_rng_key(rngs, "params", fold_index=0)
            s_meta = dict(scale_metadata) if scale_metadata is not None else {}
            self.scale = self.register_variable(
                "scale",
                Param(s_fn(s_key, (self.num_features,), param_dtype), **s_meta),
            )
        else:
            self.scale = None

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, zeros_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, (self.num_features,), param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        use_running_average: Optional[bool] = None,
        *,
        mask: Optional[ArrayLike] = None,
    ) -> JaxArray:
        r"""Normalize ``x`` using either batch statistics or stored running statistics.

        Args:
            x: Input tensor to normalize.
            use_running_average: Optional override for ``self.use_running_average``.
            mask: Optional boolean mask for masked batch statistics.

        Returns:
            Normalized ``JaxArray`` of the same shape as ``x``.
        """
        arr = ensure_array(x, dtype=self.dtype)
        s_raw: Optional[JaxArray] = (
            ensure_array(self.scale.value, dtype=self.dtype)
            if (self.use_scale and self.scale is not None)
            else None
        )
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype)
            if (self.use_bias and self.bias is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(arr, s_raw, b_raw, dtype=self.dtype)
            arr = promoted[0]
            if s_raw is not None:
                s_raw = promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]

        use_ra = (
            use_running_average
            if use_running_average is not None
            else bool(self.use_running_average)
        )
        feat_axis = self.axis if self.axis >= 0 else max(0, arr.ndim + self.axis)
        _assert_feature_dim(arr, feat_axis, self.num_features)
        reduction_axes = tuple(i for i in range(arr.ndim) if i != feat_axis)

        if use_ra:
            mean_val = ensure_array(self.mean.value)
            var_val = ensure_array(self.var.value)
        else:
            mean_val, var_val = _compute_moments(
                arr,
                reduction_axes,
                mask=mask,
                keepdims=False,
                subtract_mean=self.subtract_mean,
                use_fast_variance=bool(
                    True if self.use_fast_variance is None else self.use_fast_variance
                ),
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups,
            )
            if self.subtract_mean:
                self.mean.value = (
                    self.momentum * ensure_array(self.mean.value) + (1.0 - self.momentum) * mean_val
                )
            self.var.value = (
                self.momentum * ensure_array(self.var.value) + (1.0 - self.momentum) * var_val
            )

        bcast_shape = [1] * arr.ndim
        bcast_shape[feat_axis] = arr.shape[feat_axis]
        mean_b = jnp.reshape(mean_val, tuple(bcast_shape)) if arr.ndim > 1 else mean_val
        var_b = jnp.reshape(var_val, tuple(bcast_shape)) if arr.ndim > 1 else var_val
        centered = (arr - mean_b) if self.subtract_mean else arr
        normed: JaxArray = centered / jnp.sqrt(var_b + self.epsilon)

        if s_raw is not None:
            s_b = jnp.reshape(s_raw, tuple(bcast_shape)) if arr.ndim > 1 else s_raw
            normed = normed * s_b
        if b_raw is not None:
            b_b = jnp.reshape(b_raw, tuple(bcast_shape)) if arr.ndim > 1 else b_raw
            normed = normed + b_b
        return normed


class LayerNorm(Module):
    r"""Layer Normalization normalizing activations across feature axes for each example.

    Attributes:
        num_features: Number of features in the normalized dimension.
        epsilon: Small constant added to variance for numerical stability.
        use_bias: Whether a learnable bias vector is applied.
        use_scale: Whether a learnable scale vector is applied.
        reduction_axes: Axes over which mean and variance are computed.
        feature_axes: Axes corresponding to the learnable scale and bias parameters.
        scale: Optional learnable scale ``Param``.
        bias: Optional learnable bias ``Param``.
    """

    num_features: int
    feature_shape: Tuple[int, ...]
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    use_bias: bool
    use_scale: bool
    reduction_axes: Union[Axes, int, Sequence[int]]
    feature_axes: Union[Axes, int, Sequence[int]]
    axis_name: Optional[str]
    axis_index_groups: AxisIndexGroups
    use_fast_variance: bool
    promote_dtype: PromoteDtypeFn
    scale: Optional[Param[JaxArray]]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        num_features: Union[int, Sequence[int]],
        *,
        epsilon: float = 1e-06,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: InitializerLike = zeros_init,
        scale_init: InitializerLike = ones_init,
        reduction_axes: Union[Axes, int, Sequence[int]] = -1,
        feature_axes: Union[Axes, int, Sequence[int]] = -1,
        axis_name: Optional[str] = None,
        axis_index_groups: AxisIndexGroups = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        scale_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize a LayerNorm module.

        Args:
            num_features: Number of feature channels (or sequence of feature dimensions) to normalize.
            epsilon: Small float added to variance to avoid division by zero.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            use_bias: Whether to include a learnable additive bias parameter.
            use_scale: Whether to include a learnable multiplicative scale parameter.
            bias_init: Initializer for the bias parameter.
            scale_init: Initializer for the scale parameter.
            reduction_axes: Axis or axes over which mean and variance are computed.
            feature_axes: Axis or axes of the learnable scale and bias parameters.
            axis_name: Optional parallel axis name.
            axis_index_groups: Optional groups of axis indices.
            use_fast_variance: Whether to use the fast variance formula.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            bias_metadata: Metadata attached to ``bias``.
            scale_metadata: Metadata attached to ``scale``.
        """
        super().__init__()
        if isinstance(num_features, int):
            self.num_features = int(num_features)
            self.feature_shape = (self.num_features,)
        else:
            self.feature_shape = tuple(int(f) for f in num_features)
            self.num_features = self.feature_shape[-1] if len(self.feature_shape) > 0 else 1
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = bool(use_bias)
        self.use_scale = bool(use_scale)
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = bool(use_fast_variance)
        self.promote_dtype = promote_dtype

        if self.use_scale:
            s_fn = resolve_initializer(scale_init, ones_init)
            s_key = resolve_rng_key(rngs, "params", fold_index=0)
            s_meta = dict(scale_metadata) if scale_metadata is not None else {}
            self.scale = self.register_variable(
                "scale",
                Param(s_fn(s_key, self.feature_shape, param_dtype), **s_meta),
            )
        else:
            self.scale = None

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, zeros_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, self.feature_shape, param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *,
        mask: Optional[ArrayLike] = None,
    ) -> JaxArray:
        r"""Apply Layer Normalization across ``reduction_axes`` of ``x``.

        Args:
            x: Input tensor to normalize.
            mask: Optional boolean mask for masked normalization.

        Returns:
            Layer-normalized ``JaxArray`` of the same shape as ``x``.
        """
        arr = ensure_array(x, dtype=self.dtype)
        s_raw: Optional[JaxArray] = (
            ensure_array(self.scale.value, dtype=self.dtype)
            if (self.use_scale and self.scale is not None)
            else None
        )
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype)
            if (self.use_bias and self.bias is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(arr, s_raw, b_raw, dtype=self.dtype)
            arr = promoted[0]
            if s_raw is not None:
                s_raw = promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]

        red_axes = _normalize_axes(self.reduction_axes, arr.ndim)
        mean, var = _compute_moments(
            arr,
            red_axes,
            mask=mask,
            keepdims=True,
            subtract_mean=True,
            use_fast_variance=self.use_fast_variance,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
        )
        normed: JaxArray = (arr - mean) / jnp.sqrt(var + self.epsilon)

        feat_axes = _normalize_axes(self.feature_axes, arr.ndim)
        for ax, exp_dim in zip(feat_axes, self.feature_shape):
            _assert_feature_dim(arr, ax, exp_dim)
        bcast_shape = [1] * arr.ndim
        for ax in feat_axes:
            bcast_shape[ax] = arr.shape[ax]

        if s_raw is not None:
            s_b = jnp.reshape(s_raw, tuple(bcast_shape)) if arr.ndim > 1 else s_raw
            normed = normed * s_b
        if b_raw is not None:
            b_b = jnp.reshape(b_raw, tuple(bcast_shape)) if arr.ndim > 1 else b_raw
            normed = normed + b_b
        return normed


class RMSNorm(Module):
    r"""Root Mean Square Layer Normalization ($y = \frac{x}{\sqrt{	ext{mean}(x^2) + \epsilon}} \cdot \gamma$).

    Attributes:
        num_features: Number of features in the normalized dimension.
        epsilon: Small constant added to mean square for numerical stability.
        use_scale: Whether a learnable scale parameter is applied.
        reduction_axes: Axes over which the mean square is computed.
        feature_axes: Axes corresponding to the scale parameter.
        scale: Optional learnable scale ``Param``.
    """

    num_features: int
    feature_shape: Tuple[int, ...]
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    use_scale: bool
    reduction_axes: Union[Axes, int, Sequence[int]]
    feature_axes: Union[Axes, int, Sequence[int]]
    axis_name: Optional[str]
    axis_index_groups: AxisIndexGroups
    use_fast_variance: bool
    promote_dtype: PromoteDtypeFn
    scale: Optional[Param[JaxArray]]

    def __init__(
        self,
        num_features: Union[int, Sequence[int]],
        *,
        epsilon: float = 1e-06,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        use_scale: bool = True,
        scale_init: InitializerLike = ones_init,
        reduction_axes: Union[Axes, int, Sequence[int]] = -1,
        feature_axes: Union[Axes, int, Sequence[int]] = -1,
        axis_name: Optional[str] = None,
        axis_index_groups: AxisIndexGroups = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        scale_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize an RMSNorm module.

        Args:
            num_features: Size of the normalized feature dimension (or sequence of dimensions).
            epsilon: Small float added inside the square root for stability.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            use_scale: Whether to multiply by a learnable scale vector.
            scale_init: Initializer for the scale vector.
            reduction_axes: Axis or axes over which root-mean-square is computed.
            feature_axes: Axis or axes corresponding to ``scale``.
            axis_name: Optional parallel axis name.
            axis_index_groups: Optional groups of axis indices.
            use_fast_variance: Whether to use fast variance computation.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            scale_metadata: Metadata mapping attached to ``scale``.
        """
        super().__init__()
        if isinstance(num_features, int):
            self.num_features = int(num_features)
            self.feature_shape = (self.num_features,)
        else:
            self.feature_shape = tuple(int(f) for f in num_features)
            self.num_features = self.feature_shape[-1] if len(self.feature_shape) > 0 else 1
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_scale = bool(use_scale)
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = bool(use_fast_variance)
        self.promote_dtype = promote_dtype

        if self.use_scale:
            s_fn = resolve_initializer(scale_init, ones_init)
            s_key = resolve_rng_key(rngs, "params", fold_index=0)
            s_meta = dict(scale_metadata) if scale_metadata is not None else {}
            self.scale = self.register_variable(
                "scale",
                Param(s_fn(s_key, self.feature_shape, param_dtype), **s_meta),
            )
        else:
            self.scale = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *,
        mask: Optional[ArrayLike] = None,
    ) -> JaxArray:
        r"""Apply Root Mean Square Normalization to ``x``.

        Args:
            x: Input tensor to normalize.
            mask: Optional boolean mask.

        Returns:
            RMS-normalized ``JaxArray`` of the same shape as ``x``.
        """
        arr = ensure_array(x, dtype=self.dtype)
        s_raw: Optional[JaxArray] = (
            ensure_array(self.scale.value, dtype=self.dtype)
            if (self.use_scale and self.scale is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(arr, s_raw, dtype=self.dtype)
            arr = promoted[0]
            if s_raw is not None:
                s_raw = promoted[1]

        red_axes = _normalize_axes(self.reduction_axes, arr.ndim)
        _, mean_sq = _compute_moments(
            arr,
            red_axes,
            mask=mask,
            keepdims=True,
            subtract_mean=False,
            use_fast_variance=self.use_fast_variance,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
        )
        normed: JaxArray = arr / jnp.sqrt(mean_sq + self.epsilon)
        if s_raw is not None:
            feat_axes = _normalize_axes(self.feature_axes, arr.ndim)
            for ax, exp_dim in zip(feat_axes, self.feature_shape):
                _assert_feature_dim(arr, ax, exp_dim)
            bcast_shape = [1] * arr.ndim
            for ax in feat_axes:
                bcast_shape[ax] = arr.shape[ax]
            s_b = jnp.reshape(s_raw, tuple(bcast_shape)) if arr.ndim > 1 else s_raw
            normed = normed * s_b
        return normed


class GroupNorm(Module):
    r"""Group Normalization dividing channels into groups and normalizing within each group.

    Attributes:
        num_features: Total number of feature channels.
        num_groups: Number of channel groups.
        group_size: Number of channels per group.
        epsilon: Small constant added to variance for stability.
        use_bias: Whether a learnable bias vector is applied.
        use_scale: Whether a learnable scale vector is applied.
        scale: Optional learnable scale ``Param``.
        bias: Optional learnable bias ``Param``.
    """

    num_features: int
    num_groups: int
    group_size: int
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    use_bias: bool
    use_scale: bool
    reduction_axes: Optional[Union[Axes, int, Sequence[int]]]
    axis_name: Optional[str]
    axis_index_groups: AxisIndexGroups
    use_fast_variance: bool
    promote_dtype: PromoteDtypeFn
    scale: Optional[Param[JaxArray]]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        num_features: int,
        num_groups: Union[int, None, _UnsetGroupSentinel] = _UNSET_GROUPS,
        group_size: Optional[int] = None,
        *,
        epsilon: float = 1e-06,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: InitializerLike = zeros_init,
        scale_init: InitializerLike = ones_init,
        reduction_axes: Optional[Union[Axes, int, Sequence[int]]] = None,
        axis_name: Optional[str] = None,
        axis_index_groups: AxisIndexGroups = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        scale_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize a GroupNorm layer.

        Args:
            num_features: Number of input feature channels.
            num_groups: Number of groups to separate the channels into. Defaults to ``32``.
            group_size: Optional explicit number of channels per group (mutually exclusive with ``num_groups``).
            epsilon: Small float added to variance for numerical stability.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            use_bias: Whether to include a learnable bias vector.
            use_scale: Whether to include a learnable scale vector.
            bias_init: Bias parameter initializer.
            scale_init: Scale parameter initializer.
            reduction_axes: Optional custom reduction axes.
            axis_name: Optional parallel axis name.
            axis_index_groups: Optional groups of axis indices.
            use_fast_variance: Whether to use fast variance computation.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            bias_metadata: Metadata mapping attached to ``bias``.
            scale_metadata: Metadata mapping attached to ``scale``.

        Raises:
            ValueError: If both ``num_groups`` and ``group_size`` are specified, if both are ``None``,
                or if ``num_features`` is not divisible by ``num_groups`` / ``group_size``.
        """
        super().__init__()
        self.num_features = int(num_features)
        explicit_ng = not isinstance(num_groups, _UnsetGroupSentinel) and num_groups is not None
        if explicit_ng and group_size is not None:
            raise ValueError("Cannot specify both 'num_groups' and 'group_size' in GroupNorm.")
        if group_size is not None:
            gs = int(group_size)
            if gs <= 0 or self.num_features % gs != 0:
                raise ValueError(
                    f"num_features ({self.num_features}) must be divisible by group_size ({gs})."
                )
            self.group_size = gs
            self.num_groups = self.num_features // gs
        else:
            if num_groups is None:
                raise ValueError("Either 'num_groups' or 'group_size' must be specified.")
            ng = 32 if isinstance(num_groups, _UnsetGroupSentinel) else int(num_groups)
            if ng <= 0 or self.num_features % ng != 0:
                raise ValueError(
                    f"num_features ({self.num_features}) must be divisible by num_groups ({ng})."
                )
            self.num_groups = ng
            self.group_size = self.num_features // ng

        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = bool(use_bias)
        self.use_scale = bool(use_scale)
        self.reduction_axes = reduction_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = bool(use_fast_variance)
        self.promote_dtype = promote_dtype

        if self.use_scale:
            s_fn = resolve_initializer(scale_init, ones_init)
            s_key = resolve_rng_key(rngs, "params", fold_index=0)
            s_meta = dict(scale_metadata) if scale_metadata is not None else {}
            self.scale = self.register_variable(
                "scale",
                Param(s_fn(s_key, (self.num_features,), param_dtype), **s_meta),
            )
        else:
            self.scale = None

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, zeros_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, (self.num_features,), param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *,
        mask: Optional[ArrayLike] = None,
    ) -> JaxArray:
        r"""Apply Group Normalization to ``x``.

        Args:
            x: Input tensor of shape ``(*batch, *spatial, num_features)``.
            mask: Optional boolean mask.

        Returns:
            Group-normalized ``JaxArray`` of the same shape as ``x``.

        Raises:
            ValueError: If the last dimension of ``x`` does not match ``self.num_features``.
        """
        arr = ensure_array(x, dtype=self.dtype)
        s_raw: Optional[JaxArray] = (
            ensure_array(self.scale.value, dtype=self.dtype)
            if (self.use_scale and self.scale is not None)
            else None
        )
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype)
            if (self.use_bias and self.bias is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(arr, s_raw, b_raw, dtype=self.dtype)
            arr = promoted[0]
            if s_raw is not None:
                s_raw = promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]

        if arr.ndim == 0 or arr.shape[-1] != self.num_features:
            raise ValueError(
                f"GroupNorm input channel dimension ({arr.shape[-1] if arr.ndim > 0 else 'scalar'}) "
                f"does not match num_features ({self.num_features})."
            )
        _assert_feature_dim(arr, -1, self.num_features)
        orig_shape = arr.shape
        grouped_shape = (*orig_shape[:-1], self.num_groups, self.group_size)
        grouped = jnp.reshape(arr, grouped_shape)
        if self.reduction_axes is not None:
            red_axes = _normalize_axes(self.reduction_axes, grouped.ndim)
        else:
            spatial_and_group = tuple(range(1, grouped.ndim - 2)) + (grouped.ndim - 1,)
            red_axes = spatial_and_group if len(spatial_and_group) > 0 else (grouped.ndim - 1,)

        grouped_mask: Optional[JaxArray] = None
        if mask is not None:
            m_arr = ensure_array(mask, dtype="float32")
            m_full = jnp.zeros(orig_shape, dtype="float32") + m_arr
            grouped_mask = jnp.reshape(m_full, grouped_shape)

        mean, var = _compute_moments(
            grouped,
            red_axes,
            mask=grouped_mask,
            keepdims=True,
            subtract_mean=True,
            use_fast_variance=self.use_fast_variance,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
        )
        normed: JaxArray = jnp.reshape((grouped - mean) / jnp.sqrt(var + self.epsilon), orig_shape)

        if s_raw is not None:
            normed = normed * s_raw
        if b_raw is not None:
            normed = normed + b_raw
        return normed


class InstanceNorm(Module):
    r"""Instance Normalization normalizing each channel independently across spatial dimensions.

    Attributes:
        num_features: Number of feature channels.
        epsilon: Small constant added to variance for stability.
        use_bias: Whether a learnable bias vector is applied.
        use_scale: Whether a learnable scale vector is applied.
        feature_axes: Axes corresponding to the feature channels.
        scale: Optional learnable scale ``Param``.
        bias: Optional learnable bias ``Param``.
    """

    num_features: int
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    use_bias: bool
    use_scale: bool
    feature_axes: Union[Axes, int, Sequence[int]]
    axis_name: Optional[str]
    axis_index_groups: AxisIndexGroups
    use_fast_variance: bool
    promote_dtype: PromoteDtypeFn
    scale: Optional[Param[JaxArray]]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-06,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: InitializerLike = zeros_init,
        scale_init: InitializerLike = ones_init,
        feature_axes: Union[Axes, int, Sequence[int]] = -1,
        axis_name: Optional[str] = None,
        axis_index_groups: AxisIndexGroups = None,
        use_fast_variance: bool = True,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        scale_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize an InstanceNorm layer.

        Args:
            num_features: Number of feature channels.
            epsilon: Small float added to variance for numerical stability.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            use_bias: Whether to include a learnable bias parameter.
            use_scale: Whether to include a learnable scale parameter.
            bias_init: Bias parameter initializer.
            scale_init: Scale parameter initializer.
            feature_axes: Axis or axes corresponding to the channel dimension.
            axis_name: Optional parallel axis name.
            axis_index_groups: Optional groups of axis indices.
            use_fast_variance: Whether to use fast variance computation.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            bias_metadata: Metadata attached to ``bias``.
            scale_metadata: Metadata attached to ``scale``.
        """
        super().__init__()
        self.num_features = int(num_features)
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = bool(use_bias)
        self.use_scale = bool(use_scale)
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = bool(use_fast_variance)
        self.promote_dtype = promote_dtype

        if self.use_scale:
            s_fn = resolve_initializer(scale_init, ones_init)
            s_key = resolve_rng_key(rngs, "params", fold_index=0)
            s_meta = dict(scale_metadata) if scale_metadata is not None else {}
            self.scale = self.register_variable(
                "scale",
                Param(s_fn(s_key, (self.num_features,), param_dtype), **s_meta),
            )
        else:
            self.scale = None

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, zeros_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, (self.num_features,), param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *,
        mask: Optional[ArrayLike] = None,
    ) -> JaxArray:
        r"""Apply Instance Normalization across spatial axes of ``x``.

        In accordance with ``flax.nnx.InstanceNorm``, reduction is performed over all axes
        except axis ``0`` (batch) and ``self.feature_axes``.

        Args:
            x: Input tensor of shape ``(batch, *spatial, num_features)``.
            mask: Optional boolean mask.

        Returns:
            Instance-normalized ``JaxArray`` of the same shape as ``x``.
        """
        arr = ensure_array(x, dtype=self.dtype)
        s_raw: Optional[JaxArray] = (
            ensure_array(self.scale.value, dtype=self.dtype)
            if (self.use_scale and self.scale is not None)
            else None
        )
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype)
            if (self.use_bias and self.bias is not None)
            else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(arr, s_raw, b_raw, dtype=self.dtype)
            arr = promoted[0]
            if s_raw is not None:
                s_raw = promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]

        feat_axes = _normalize_axes(self.feature_axes, arr.ndim)
        for ax in feat_axes:
            _assert_feature_dim(arr, ax, self.num_features)
        red_axes = tuple(i for i in range(1, arr.ndim) if i not in feat_axes)
        mean, var = _compute_moments(
            arr,
            red_axes,
            mask=mask,
            keepdims=True,
            subtract_mean=True,
            use_fast_variance=self.use_fast_variance,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
        )
        normed: JaxArray = (arr - mean) / jnp.sqrt(var + self.epsilon)

        bcast_shape = [1] * arr.ndim
        for ax in feat_axes:
            bcast_shape[ax] = arr.shape[ax]

        if s_raw is not None:
            s_b = jnp.reshape(s_raw, tuple(bcast_shape)) if arr.ndim > 1 else s_raw
            normed = normed * s_b
        if b_raw is not None:
            b_b = jnp.reshape(b_raw, tuple(bcast_shape)) if arr.ndim > 1 else b_raw
            normed = normed + b_b
        return normed


def _find_first_matrix_param(layer: Module) -> Optional[AnyVariable]:
    """Find the primary weight ``Param`` inside ``layer`` (preferring ``"kernel"``).

    Args:
        layer: Target ``Module`` to inspect.

    Returns:
        Matched ``Param`` / ``Variable`` or ``None`` if no parameters exist.
    """
    layer._ensure_registries()
    if "kernel" in layer._variables:
        return layer._variables["kernel"]
    for _, node in iter_graph(layer):
        if isinstance(node, Param):
            return node
    return None


class SpectralNorm(Module):
    r"""Spectral Normalization wrapper bounding the Lipschitz constant of a wrapped layer.

    Uses power iteration to estimate the largest singular value $\sigma(W)$ of the
    wrapped layer's weight matrix and divides the weight matrix by $\sigma(W)$
    during the forward pass.

    Attributes:
        layer_instance: Wrapped ``Module`` instance containing a weight parameter.
        n_steps: Number of power iteration steps per forward pass.
        epsilon: Small constant for vector normalization stability.
        error_on_non_matrix: Whether to raise a ``ValueError`` if a parameter is 1D/scalar.
        update_stats: Default flag controlling whether power iteration updates ``u`` and ``v``.
        u: Left singular vector estimate stored as a ``BatchStat``.
        v: Right singular vector estimate stored as a ``BatchStat``.
    """

    layer_instance: Module
    n_steps: int
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    error_on_non_matrix: bool
    update_stats: bool
    u: Optional[BatchStat[JaxArray]]
    v: Optional[BatchStat[JaxArray]]

    def __init__(
        self,
        layer_instance: Module,
        *,
        n_steps: int = 1,
        epsilon: float = 1e-12,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        error_on_non_matrix: bool = False,
        update_stats: bool = True,
        rngs: Optional[Union[Rngs, RngStream]] = None,
    ) -> None:
        r"""Initialize a SpectralNorm wrapper around ``layer_instance``.

        Args:
            layer_instance: The target ``Module`` (e.g., ``Linear`` or ``Conv``) to wrap.
            n_steps: Number of power iteration updates per call. Defaults to ``1``.
            epsilon: Small float for L2 normalization stability.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            error_on_non_matrix: Raise ``ValueError`` if the target parameter has fewer than 2 dims.
            update_stats: Whether to update singular vector estimates ``u`` and ``v`` by default.
            rngs: Random number generator stream(s).

        Raises:
            ValueError: If ``n_steps < 1`` or ``error_on_non_matrix=True`` on a 1D parameter.
        """
        super().__init__()
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        self.layer_instance = self.register_child("layer_instance", layer_instance)
        self.n_steps = int(n_steps)
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.error_on_non_matrix = bool(error_on_non_matrix)
        self.update_stats = bool(update_stats)

        kernel_var = _find_first_matrix_param(self.layer_instance)
        if kernel_var is not None:
            k_arr = ensure_array(kernel_var.value, dtype=param_dtype)
            if k_arr.ndim < 2 and self.error_on_non_matrix:
                raise ValueError("SpectralNorm requires a weight tensor with ndim >= 2.")
            rows = int(k_arr.shape[0]) if k_arr.ndim >= 1 else 1
            cols = 1
            for d in k_arr.shape[1:]:
                cols *= int(d)
            if rngs is not None:
                u_key = resolve_rng_key(rngs, "params", fold_index=0)
                v_key = resolve_rng_key(rngs, "params", fold_index=1)
                u_raw = jrandom.normal(u_key, (rows,), dtype=param_dtype)
                v_raw = jrandom.normal(v_key, (cols,), dtype=param_dtype)
                u_init = u_raw / jnp.sqrt(jnp.sum(u_raw * u_raw) + self.epsilon)
                v_init = v_raw / jnp.sqrt(jnp.sum(v_raw * v_raw) + self.epsilon)
            else:
                u_init = ensure_array(jnp.ones((rows,)) / (rows**0.5), dtype=param_dtype)
                v_init = ensure_array(jnp.ones((cols,)) / (cols**0.5), dtype=param_dtype)
            self.u = self.register_variable("u", BatchStat(u_init))
            self.v = self.register_variable("v", BatchStat(v_init))
        else:
            self.u = None
            self.v = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *,
        update_stats: Optional[bool] = None,
    ) -> JaxArray:
        r"""Apply spectral normalization to ``layer_instance``'s kernel and evaluate ``layer_instance(x)``.

        Args:
            x: Input tensor passed to ``self.layer_instance``.
            update_stats: Optional override for ``self.update_stats``.

        Returns:
            Output ``JaxArray`` from ``self.layer_instance(x)``.
        """
        do_update = self.update_stats if update_stats is None else bool(update_stats)
        kernel_var = _find_first_matrix_param(self.layer_instance)
        if kernel_var is not None and self.u is not None and self.v is not None:
            orig_k = ensure_array(kernel_var.value, dtype=self.dtype)
            rows = orig_k.shape[0]
            w_mat = jnp.reshape(orig_k, (rows, -1))
            u_vec = ensure_array(self.u.value, dtype=self.dtype)
            v_vec = ensure_array(self.v.value, dtype=self.dtype)
            for _ in range(self.n_steps):
                v_raw = jnp.dot(jnp.transpose(w_mat), u_vec)
                v_norm = jnp.sqrt(jnp.sum(v_raw * v_raw) + self.epsilon)
                v_vec = v_raw / v_norm
                u_raw = jnp.dot(w_mat, v_vec)
                u_norm = jnp.sqrt(jnp.sum(u_raw * u_raw) + self.epsilon)
                u_vec = u_raw / u_norm
            if do_update:
                self.u.value = u_vec
                self.v.value = v_vec
            sigma = jnp.sum(u_vec * jnp.dot(w_mat, v_vec))
            kernel_var.value = orig_k / jnp.maximum(sigma, self.epsilon)
            try:
                out = _invoke_wrapped_layer(self.layer_instance, x)
            finally:
                kernel_var.value = orig_k
            return out

        return _invoke_wrapped_layer(self.layer_instance, x)


_DEFAULT_KERNEL_FILTER: PathContains = PathContains("kernel")


class WeightNorm(Module):
    r"""Weight Normalization wrapper decoupling weight magnitude and direction ($w = g \frac{v}{\|v\|}$).

    Attributes:
        layer_instance: Wrapped ``Module`` instance.
        feature_axes: Axis or axes kept separate when computing parameter L2 norms.
        use_scale: Whether to multiply normalized weights by a learnable direction scale ``g``.
        epsilon: Small constant added to the L2 norm for numerical stability.
        variable_filter: Filter selecting which parameter(s) of ``layer_instance`` to normalize.
        scale: Primary learnable magnitude ``Param`` ``g`` (for the first matched parameter).
        scales: Dictionary mapping matched variable path keys to their respective ``Param`` scales.
    """

    layer_instance: Module
    feature_axes: Optional[Union[Axes, int, Sequence[int]]]
    use_scale: bool
    epsilon: float
    dtype: Optional[DType]
    param_dtype: DType
    variable_filter: FilterLiteral
    promote_dtype: PromoteDtypeFn
    scale: Optional[Param[JaxArray]]
    scales: dict[str, Param[JaxArray]]

    def __init__(
        self,
        layer_instance: Module,
        *,
        feature_axes: Optional[Union[Axes, int, Sequence[int]]] = -1,
        use_scale: bool = True,
        scale_init: InitializerLike = ones_init,
        epsilon: float = 1e-12,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        variable_filter: FilterLiteral = _DEFAULT_KERNEL_FILTER,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
    ) -> None:
        r"""Initialize a WeightNorm wrapper around ``layer_instance``.

        Args:
            layer_instance: Target ``Module`` whose weights will be normalized.
            feature_axes: Output feature axis or axes along which separate norms are computed.
            use_scale: Whether to include a learnable magnitude parameter ``scale``.
            scale_init: Initializer for the magnitude parameter ``scale``.
            epsilon: Small float added to the weight L2 norm.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            variable_filter: Filter predicate selecting weight variables (defaults to ``PathContains("kernel")``).
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
        """
        super().__init__()
        self.layer_instance = self.register_child("layer_instance", layer_instance)
        self.feature_axes = feature_axes
        self.use_scale = bool(use_scale)
        self.epsilon = float(epsilon)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.variable_filter = variable_filter
        self.promote_dtype = promote_dtype
        self.scales = {}
        self.scale = None

        self.layer_instance._ensure_registries()
        predicate = to_predicate(self.variable_filter)
        s_fn = resolve_initializer(scale_init, ones_init)
        fold_idx = 0
        for path, node in iter_graph(self.layer_instance):
            if isinstance(node, Param) and predicate(path, node) and self.use_scale:
                path_key = ".".join(path)
                k_arr = ensure_array(node.value)
                f_axes = _normalize_axes(self.feature_axes, k_arr.ndim)
                scale_shape = (
                    tuple(int(k_arr.shape[ax]) for ax in f_axes) if k_arr.ndim >= 1 else (1,)
                )
                s_key = resolve_rng_key(rngs, "params", fold_index=fold_idx)
                fold_idx += 1
                reg_name = "scale" if len(self.scales) == 0 else f"scale_{'_'.join(path)}"
                s_param = self.register_variable(
                    reg_name,
                    Param(s_fn(s_key, scale_shape, param_dtype)),
                )
                self.scales[path_key] = s_param
                if len(path) == 1:
                    self.scales[path[0]] = s_param
                if self.scale is None:
                    self.scale = s_param
        self.finish_init()

    def __call__(self, x: ArrayLike) -> JaxArray:
        r"""Normalize the filtered weight parameter(s) of ``layer_instance`` and evaluate ``layer_instance(x)``.

        Args:
            x: Input tensor passed to ``self.layer_instance``.

        Returns:
            Output ``JaxArray`` from ``self.layer_instance(x)``.
        """
        self.layer_instance._ensure_registries()
        predicate = to_predicate(self.variable_filter)
        saved: list[Tuple[Param[JaxArray], JaxArray]] = []
        for path, node in iter_graph(self.layer_instance):
            if isinstance(node, Param) and predicate(path, node):
                path_key = ".".join(path)
                orig_w = ensure_array(node.value)
                saved.append((node, orig_w))
                w_work = ensure_array(orig_w, dtype=self.dtype)
                s_param = self.scales.get(path_key, self.scale)
                s_val: Optional[JaxArray] = (
                    ensure_array(s_param.value, dtype=self.dtype)
                    if (self.use_scale and s_param is not None)
                    else None
                )
                if self.promote_dtype is not None:
                    promoted = self.promote_dtype(w_work, s_val, dtype=self.dtype)
                    w_work = promoted[0]
                    if s_val is not None:
                        s_val = promoted[1]

                f_axes = _normalize_axes(self.feature_axes, w_work.ndim)
                red_axes = tuple(i for i in range(w_work.ndim) if i not in f_axes)
                norm = jnp.sqrt(
                    jnp.sum(w_work * w_work, axis=red_axes, keepdims=True) + self.epsilon
                )
                normed_w = w_work / norm
                if s_val is not None:
                    if w_work.ndim > 1:
                        bcast_shape = [1] * w_work.ndim
                        for ax in f_axes:
                            bcast_shape[ax] = w_work.shape[ax]
                        s_b = jnp.reshape(s_val, tuple(bcast_shape))
                    else:
                        s_b = s_val
                    normed_w = normed_w * s_b
                node.value = normed_w

        try:
            out = _invoke_wrapped_layer(self.layer_instance, x)
        finally:
            for param_node, orig_w in saved:
                param_node.value = orig_w
        return out


__all__ = [
    "BatchNorm",
    "GroupNorm",
    "InstanceNorm",
    "LayerNorm",
    "RMSNorm",
    "SpectralNorm",
    "WeightNorm",
]
