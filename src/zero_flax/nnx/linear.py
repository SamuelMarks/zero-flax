"""Linear, Dense, LinearGeneral, Einsum, LoRA, and LoRALinear layers for zero-flax NNX.

This module implements standard and generalized linear transformations and Low-Rank
Adaptation (LoRA) layers with exact parameter signatures matching `flax_nnx_v0.12.10`
and full mathematical forward-pass execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Optional, Tuple, Union

from zero_jax import lax
from zero_jax import numpy as jnp

from zero_flax.nnx.module import Module
from zero_flax.nnx.rnglib import Rngs, RngStream, resolve_rng_key
from zero_flax.nnx.variables import LoRAParam, Param, Variable
from zero_flax.types import (
    EMPTY_METADATA,
    ArrayLike,
    DotGeneralLike,
    DotGeneralProtocol,
    DotGeneralT,
    DType,
    EinsumOpT,
    InitializerLike,
    JaxArray,
    MetadataMapping,
    MetadataValue,
    PrecisionSpec,
    PromoteDtypeFn,
    Shape,
    default_bias_init,
    default_kernel_init,
    default_promote_dtype,
    ensure_array,
    resolve_initializer,
)


def _assert_axis_dimension(tensor: JaxArray, axis: int, expected: int) -> None:
    """Validate that ``tensor`` has rank >= 1 and size ``expected`` along ``axis`` via ``zero_chex``.

    Args:
        tensor: Input ``JaxArray`` to validate.
        axis: Axis index to check.
        expected: Expected dimension size along ``axis``.
    """
    from zero_chex.assertions import assert_axis_dimension, assert_rank

    assert_rank(tensor, tensor.ndim)
    assert_axis_dimension(tensor, axis, expected)


def _normalize_einsum_equation(einsum_str: str) -> str:
    """Normalize an Einstein summation equation, inferring output subscripts if omitted.

    Supports both standard subscript equations (e.g., ``"ab,bc"`` -> ``"ab,bc->ac"``),
    ellipsis broadcasting equations across one or both operands (e.g., ``"...ab,...bc"`` ->
    ``"...ab,...bc->...ac"``), and trace/diagonal subscripts within an operand.

    Args:
        einsum_str: Raw Einstein summation string (e.g., ``"ab,bc->ac"`` or ``"ab,bc"``).

    Returns:
        Normalized equation string containing explicit ``"->"`` output subscripts.

    Raises:
        ValueError: If ``einsum_str`` is empty or lacks a comma separating operands.
    """
    cleaned = einsum_str.replace(" ", "")
    if "->" in cleaned:
        return cleaned
    if "," not in cleaned:
        raise ValueError(f"Invalid einsum equation {einsum_str!r}: missing ',' between operands.")
    operands = cleaned.split(",")
    has_ellipsis = any("..." in op for op in operands)
    stripped_operands = "".join(op.replace("...", "") for op in operands)
    total_counts: dict[str, int] = {}
    for ch in stripped_operands:
        total_counts[ch] = total_counts.get(ch, 0) + 1
    out_chars = "".join(sorted(ch for ch, cnt in total_counts.items() if cnt == 1))
    prefix = "..." if has_ellipsis else ""
    return f"{cleaned}->{prefix}{out_chars}"


class Linear(Module):
    """Standard dense linear transformation layer ($y = x W + b$).

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether a learnable bias vector is added.
        dtype: Computation data type.
        param_dtype: Parameter initialization data type.
        precision: Numerical precision specification.
        kernel_init: Initializer for the weight matrix.
        bias_init: Initializer for the bias vector.
        dot_general: Generalized dot product function.
        promote_dtype: Dtype promotion callable.
        preferred_element_type: Optional preferred accumulation dtype.
        kernel: Learnable weight matrix ``Param`` of shape ``(in_features, out_features)``.
        bias: Optional learnable bias vector ``Param`` of shape ``(out_features,)``.
    """

    in_features: int
    out_features: int
    use_bias: bool
    dtype: Optional[DType]
    param_dtype: DType
    precision: PrecisionSpec
    kernel_init: InitializerLike
    bias_init: InitializerLike
    dot_general: DotGeneralLike
    promote_dtype: PromoteDtypeFn
    preferred_element_type: Optional[DType]
    kernel: Param[JaxArray]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        precision: PrecisionSpec = None,
        kernel_init: InitializerLike = default_kernel_init,
        bias_init: InitializerLike = default_bias_init,
        dot_general: DotGeneralLike = None,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        preferred_element_type: Optional[DType] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize a Linear layer.

        Args:
            in_features: Size of the last dimension of the input tensor.
            out_features: Size of the last dimension of the output tensor.
            use_bias: Whether to add a learnable bias vector. Defaults to ``True``.
            dtype: Data type of the computation.
            param_dtype: Data type of the initialized parameters.
            precision: Numerical precision for the matrix multiplication.
            kernel_init: Initializer function for the weight matrix.
            bias_init: Initializer function for the bias vector.
            dot_general: Generalized contraction function.
            promote_dtype: Function used to promote input and parameter dtypes.
            preferred_element_type: Optional accumulation data type.
            rngs: Random number generator stream(s) for parameter initialization.
            kernel_metadata: Optional metadata mapping attached to ``kernel``.
            bias_metadata: Optional metadata mapping attached to ``bias``.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general
        self.promote_dtype = promote_dtype
        self.preferred_element_type = preferred_element_type

        k_fn = resolve_initializer(kernel_init, default_kernel_init)
        k_key = resolve_rng_key(rngs, "params", fold_index=0)
        k_meta = dict(kernel_metadata) if kernel_metadata is not None else {}
        self.kernel = self.register_variable(
            "kernel",
            Param(k_fn(k_key, (self.in_features, self.out_features), param_dtype), **k_meta),
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
        """Apply the linear transformation $y = x W + b$ to ``inputs``.

        Args:
            inputs: Input tensor whose last dimension has size ``in_features``.

        Returns:
            Transformed output ``JaxArray`` with last dimension ``out_features``.

        Raises:
            ValueError: If the last dimension of ``inputs`` does not match ``self.in_features``.
        """
        x = ensure_array(inputs)
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear input last dimension ({x.shape[-1] if x.ndim > 0 else 'scalar'}) "
                f"does not match in_features ({self.in_features})."
            )
        _assert_axis_dimension(x, -1, self.in_features)
        k = ensure_array(self.kernel.value, dtype=self.dtype)
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
        if self.dot_general is not None:
            last_axis = x.ndim - 1
            out: JaxArray = self.dot_general(
                x,
                k,
                (((last_axis,), (0,)), ((), ())),
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
            )
        else:
            out = jnp.dot(x, k)
        if b_raw is not None:
            out = out + b_raw
        if self.preferred_element_type is not None:
            out = ensure_array(out, dtype=self.preferred_element_type)
        return out


class Dense(Linear):
    """Dense linear layer alias inheriting full functionality from ``Linear``.

    Attributes:
        features: Number of output features (alias for ``out_features``).
    """

    features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        precision: PrecisionSpec = None,
        kernel_init: InitializerLike = default_kernel_init,
        bias_init: InitializerLike = default_bias_init,
        dot_general: DotGeneralLike = None,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        preferred_element_type: Optional[DType] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize a Dense layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            use_bias: Whether to include a learnable bias vector.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            precision: Numerical precision specification.
            kernel_init: Weight matrix initializer.
            bias_init: Bias vector initializer.
            dot_general: Generalized dot product callable.
            promote_dtype: Dtype promotion callable.
            preferred_element_type: Optional accumulation dtype.
            rngs: Random number generator streams.
            kernel_metadata: Metadata attached to ``kernel``.
            bias_metadata: Metadata attached to ``bias``.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dot_general=dot_general,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
        )
        self.features = self.out_features


class LinearGeneral(Module):
    """Generalized linear transformation over arbitrary input, batch, and output feature axes.

    Attributes:
        in_features: Tuple of input feature dimension sizes.
        out_features: Tuple of output feature dimension sizes.
        axis: Tuple of input axes to contract.
        batch_axis: Mapping of input batch axes to kernel batch axes.
        batch_dims: Tuple of batch dimension sizes initialized in ``kernel`` and ``bias``.
        use_bias: Whether to add a learnable bias tensor.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        kernel: Learnable weight tensor ``Param`` of shape ``(*batch_dims, *in_features, *out_features)``.
        bias: Optional learnable bias tensor ``Param`` of shape ``(*batch_dims, *out_features)``.
    """

    in_features: Tuple[int, ...]
    out_features: Tuple[int, ...]
    axis: Tuple[int, ...]
    batch_axis: dict[int, int]
    batch_dims: Tuple[int, ...]
    use_bias: bool
    dtype: Optional[DType]
    param_dtype: DType
    precision: PrecisionSpec
    promote_dtype: PromoteDtypeFn
    dot_general: DotGeneralLike
    dot_general_cls: Optional[type[DotGeneralT]]
    preferred_element_type: Optional[DType]
    kernel: Param[JaxArray]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        in_features: Union[int, Sequence[int]],
        out_features: Union[int, Sequence[int]],
        *,
        axis: Union[int, Sequence[int]] = -1,
        batch_axis: Union[Mapping[int, int], Sequence[int], None] = MappingProxyType({}),
        use_bias: bool = True,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        kernel_init: InitializerLike = default_kernel_init,
        bias_init: InitializerLike = default_bias_init,
        precision: PrecisionSpec = None,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        dot_general: DotGeneralLike = None,
        dot_general_cls: Optional[type[DotGeneralT]] = None,
        preferred_element_type: Optional[DType] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize a LinearGeneral layer.

        Args:
            in_features: Input feature dimension size or sequence of sizes.
            out_features: Output feature dimension size or sequence of sizes.
            axis: Input axis or sequence of axes along which to contract.
            batch_axis: Optional mapping ``(input_axis -> batch_dim_size)`` or sequence of batch axes.
            use_bias: Whether to include a learnable bias tensor.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            kernel_init: Initializer for the weight tensor.
            bias_init: Initializer for the bias tensor.
            precision: Numerical precision specification.
            promote_dtype: Dtype promotion callable.
            dot_general: Optional generalized contraction callable.
            dot_general_cls: Optional class factory for ``dot_general``.
            preferred_element_type: Optional accumulation dtype.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata attached to ``kernel``.
            bias_metadata: Metadata attached to ``bias``.

        Raises:
            ValueError: If ``len(axis) != len(in_features)``.
        """
        super().__init__()
        self.in_features = (
            (int(in_features),)
            if isinstance(in_features, int)
            else tuple(int(f) for f in in_features)
        )
        self.out_features = (
            (int(out_features),)
            if isinstance(out_features, int)
            else tuple(int(f) for f in out_features)
        )
        if isinstance(axis, int):
            if len(self.in_features) > 1:
                self.axis = tuple(range(-len(self.in_features), 0))
            else:
                self.axis = (int(axis),)
        else:
            self.axis = tuple(int(a) for a in axis)
        if len(self.axis) != len(self.in_features):
            raise ValueError(
                f"LinearGeneral axis length ({len(self.axis)}) must match "
                f"in_features length ({len(self.in_features)})."
            )
        if batch_axis is None:
            self.batch_axis = {}
            self.batch_dims = ()
        elif isinstance(batch_axis, Mapping):
            self.batch_axis = {int(k): idx for idx, k in enumerate(batch_axis.keys())}
            self.batch_dims = tuple(max(1, int(v)) for v in batch_axis.values())
        else:
            self.batch_axis = {int(ax): idx for idx, ax in enumerate(batch_axis)}
            self.batch_dims = (1,) * len(self.batch_axis)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.promote_dtype = promote_dtype
        self.dot_general = dot_general
        self.dot_general_cls = dot_general_cls
        self.preferred_element_type = preferred_element_type

        k_fn = resolve_initializer(kernel_init, default_kernel_init)
        k_key = resolve_rng_key(rngs, "params", fold_index=0)
        k_meta = dict(kernel_metadata) if kernel_metadata is not None else {}
        kernel_shape = self.batch_dims + self.in_features + self.out_features
        self.kernel = self.register_variable(
            "kernel",
            Param(k_fn(k_key, kernel_shape, param_dtype), **k_meta),
        )

        if self.use_bias:
            b_fn = resolve_initializer(bias_init, default_bias_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            bias_shape = self.batch_dims + self.out_features
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, bias_shape, param_dtype), **b_meta),
            )
        else:
            self.bias = None
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        """Contract ``inputs`` along ``self.axis`` with ``self.kernel`` and add ``self.bias``.

        Args:
            inputs: Input array whose ``self.axis`` dimensions match ``self.in_features``.

        Returns:
            Contracted output ``JaxArray`` with trailing dimensions ``self.out_features``.

        Raises:
            ValueError: If contracted input dimensions do not match ``self.in_features``.
        """
        x = ensure_array(inputs)
        k = ensure_array(self.kernel.value, dtype=self.dtype)
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

        ndim = x.ndim
        norm_axes = tuple(a if a >= 0 else ndim + a for a in self.axis)
        lhs_batch_axes = tuple(a if a >= 0 else ndim + a for a in self.batch_axis)
        rhs_batch_axes = tuple(range(len(self.batch_axis)))
        if len(rhs_batch_axes) > 0:
            expanded_k = k
            for lhs_ax, rhs_ax in zip(lhs_batch_axes, rhs_batch_axes):
                if expanded_k.shape[rhs_ax] == 1 and x.shape[lhs_ax] > 1:
                    expanded_k = jnp.concatenate(
                        [expanded_k for _ in range(x.shape[lhs_ax])], axis=rhs_ax
                    )
            k = expanded_k
        kernel_contract_axes = tuple(
            range(len(rhs_batch_axes), len(rhs_batch_axes) + len(self.in_features))
        )
        for ax, expected_dim in zip(norm_axes, self.in_features):
            if ax < 0 or ax >= ndim or x.shape[ax] != expected_dim:
                raise ValueError(
                    f"LinearGeneral input dimension at axis {ax} ({x.shape[ax] if 0 <= ax < ndim else 'out of bounds'}) "
                    f"does not match expected in_features {expected_dim}."
                )
            _assert_axis_dimension(x, ax, expected_dim)

        dg: Optional[Union[DotGeneralProtocol, DotGeneralT]] = None
        if self.dot_general is not None:
            dg = self.dot_general
        elif self.dot_general_cls is not None:
            dg = self.dot_general_cls()
        elif len(lhs_batch_axes) > 0:
            dg = DotGeneralT()

        if dg is not None:
            out: JaxArray = dg(
                x,
                k,
                ((norm_axes, kernel_contract_axes), (lhs_batch_axes, rhs_batch_axes)),
                precision=self.precision,
                preferred_element_type=self.preferred_element_type,
            )
        else:
            out = jnp.tensordot(x, k, axes=(norm_axes, kernel_contract_axes))

        if b_raw is not None:
            b_arr = b_raw
            if b_arr.ndim < out.ndim:
                bcast_shape = (
                    b_arr.shape[: len(rhs_batch_axes)]
                    + (1,) * (out.ndim - b_arr.ndim)
                    + b_arr.shape[len(rhs_batch_axes) :]
                )
                b_arr = jnp.reshape(b_arr, bcast_shape)
            out = out + b_arr
        if self.preferred_element_type is not None:
            out = ensure_array(out, dtype=self.preferred_element_type)
        return out


class Einsum(Module):
    """Learnable Einstein summation layer with parameterized kernel and optional bias.

    Attributes:
        einsum_str: Normalized Einstein summation subscript string.
        kernel_shape: Tuple of kernel dimensions.
        bias_shape: Optional tuple of bias dimensions.
        kernel: Learnable weight ``Param``.
        bias: Optional learnable bias ``Param``.
    """

    einsum_str: str
    kernel_shape: Tuple[int, ...]
    bias_shape: Optional[Tuple[int, ...]]
    dtype: Optional[DType]
    param_dtype: DType
    precision: PrecisionSpec
    promote_dtype: PromoteDtypeFn
    einsum_op: EinsumOpT
    preferred_element_type: Optional[DType]
    kernel: Param[JaxArray]
    bias: Optional[Param[JaxArray]]

    def __init__(
        self,
        einsum_str: str,
        kernel_shape: Union[Shape, Sequence[int]],
        bias_shape: Optional[Union[Shape, Sequence[int]]] = None,
        *,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        precision: PrecisionSpec = None,
        kernel_init: InitializerLike = default_kernel_init,
        bias_init: InitializerLike = default_bias_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        einsum_op: EinsumOpT = None,
        preferred_element_type: Optional[DType] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize an Einsum layer.

        Args:
            einsum_str: Einstein summation equation string (e.g., ``"ab,bc->ac"`` or ``"ab,bc"``).
            kernel_shape: Shape of the learnable weight tensor.
            bias_shape: Optional shape of the learnable bias tensor.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            precision: Numerical precision specification.
            kernel_init: Weight tensor initializer.
            bias_init: Bias tensor initializer.
            promote_dtype: Dtype promotion callable.
            einsum_op: Underlying einsum operator callable.
            preferred_element_type: Optional accumulation dtype.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata attached to ``kernel``.
            bias_metadata: Metadata attached to ``bias``.
        """
        super().__init__()
        self.einsum_str = _normalize_einsum_equation(einsum_str)
        self.kernel_shape = tuple(int(d) for d in kernel_shape)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.promote_dtype = promote_dtype
        self.einsum_op = einsum_op
        self.preferred_element_type = preferred_element_type

        k_fn = resolve_initializer(kernel_init, default_kernel_init)
        k_key = resolve_rng_key(rngs, "params", fold_index=0)
        k_meta = dict(kernel_metadata) if kernel_metadata is not None else {}
        self.kernel = self.register_variable(
            "kernel",
            Param(k_fn(k_key, self.kernel_shape, param_dtype), **k_meta),
        )

        if bias_shape is not None:
            self.bias_shape = tuple(int(d) for d in bias_shape)
            b_fn = resolve_initializer(bias_init, default_bias_init)
            b_key = resolve_rng_key(rngs, "params", fold_index=1)
            b_meta = dict(bias_metadata) if bias_metadata is not None else {}
            self.bias = self.register_variable(
                "bias",
                Param(b_fn(b_key, self.bias_shape, param_dtype), **b_meta),
            )
        else:
            self.bias_shape = None
            self.bias = None
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        """Evaluate the Einstein summation between ``inputs`` and ``self.kernel``.

        Args:
            inputs: Input array matching the first operand of ``self.einsum_str``.

        Returns:
            Resulting ``JaxArray`` (plus ``self.bias`` if configured).
        """
        x = ensure_array(inputs, dtype=self.dtype)
        k = ensure_array(self.kernel.value, dtype=self.dtype)
        b_raw: Optional[JaxArray] = (
            ensure_array(self.bias.value, dtype=self.dtype) if self.bias is not None else None
        )
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(x, k, b_raw, dtype=self.dtype)
            x, k = promoted[0], promoted[1]
            if b_raw is not None:
                b_raw = promoted[2]
        if self.einsum_op is not None:
            out: JaxArray = self.einsum_op(self.einsum_str, x, k)
        else:
            out = jnp.einsum(self.einsum_str, x, k)
        if b_raw is not None:
            b_arr = b_raw
            if b_arr.ndim < out.ndim:
                b_arr = jnp.reshape(b_arr, (1,) * (out.ndim - b_arr.ndim) + b_arr.shape)
            out = out + b_arr
        if self.preferred_element_type is not None:
            out = ensure_array(out, dtype=self.preferred_element_type)
        return out


class LoRA(Module):
    """Low-Rank Adaptation (LoRA) module computing $y = (x A) B$ (plus optional base module output).

    Attributes:
        in_features: Input feature dimension size.
        lora_rank: Low-rank bottleneck dimension size.
        out_features: Output feature dimension size.
        base_module: Optional wrapped base ``Module`` whose output is added to the LoRA branch.
        lora_a: Low-rank down-projection parameter of shape ``(in_features, lora_rank)``.
        lora_b: Low-rank up-projection parameter of shape ``(lora_rank, out_features)``.
    """

    in_features: int
    lora_rank: int
    out_features: int
    base_module: Optional[Module]
    dtype: Optional[DType]
    param_dtype: DType
    precision: PrecisionSpec
    lora_param_type: type[Variable[JaxArray]]
    promote_dtype: PromoteDtypeFn
    lora_a: Variable[JaxArray]
    lora_b: Variable[JaxArray]

    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        precision: PrecisionSpec = None,
        *,
        base_module: Optional[Module] = None,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        a_initializer: InitializerLike = default_kernel_init,
        b_initializer: InitializerLike = default_bias_init,
        lora_param_type: type[Variable[JaxArray]] = LoRAParam,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        a_metadata: MetadataMapping = EMPTY_METADATA,
        b_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        """Initialize a LoRA layer.

        Args:
            in_features: Number of input features.
            lora_rank: Low-rank bottleneck rank $r$.
            out_features: Number of output features.
            precision: Optional matrix multiplication precision.
            base_module: Optional base ``Module`` evaluated in parallel with the LoRA adapter.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            a_initializer: Initializer for down-projection matrix ``lora_a``.
            b_initializer: Initializer for up-projection matrix ``lora_b``.
            lora_param_type: Variable subclass used to wrap ``lora_a`` and ``lora_b``.
            promote_dtype: Dtype promotion callable.
            rngs: Random number generator stream(s).
            a_metadata: Metadata attached to ``lora_a``.
            b_metadata: Metadata attached to ``lora_b``.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.lora_rank = int(lora_rank)
        self.out_features = int(out_features)
        self.precision = precision
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.promote_dtype = promote_dtype

        if isinstance(base_module, Module):
            self.base_module = self.register_child("base_module", base_module)
        else:
            self.base_module = None

        self.lora_param_type = lora_param_type

        a_fn = resolve_initializer(a_initializer, default_kernel_init)
        b_fn = resolve_initializer(b_initializer, default_bias_init)
        a_key = resolve_rng_key(rngs, "params", fold_index=0)
        b_key = resolve_rng_key(rngs, "params", fold_index=1)
        a_meta = dict(a_metadata) if a_metadata is not None else {}
        b_meta = dict(b_metadata) if b_metadata is not None else {}

        self.lora_a = self.register_variable(
            "lora_a",
            lora_param_type(a_fn(a_key, (self.in_features, self.lora_rank), param_dtype), **a_meta),
        )
        self.lora_b = self.register_variable(
            "lora_b",
            lora_param_type(
                b_fn(b_key, (self.lora_rank, self.out_features), param_dtype), **b_meta
            ),
        )
        self.finish_init()

    def __call__(self, x: ArrayLike) -> JaxArray:
        """Apply the low-rank adaptation $(x A) B$ (and add ``base_module(x)`` if present).

        Args:
            x: Input array with last dimension ``in_features``.

        Returns:
            Output ``JaxArray`` with last dimension ``out_features``.

        Raises:
            ValueError: If the last dimension of ``x`` does not match ``self.in_features``.
        """
        arr = ensure_array(x)
        if arr.ndim == 0 or arr.shape[-1] != self.in_features:
            raise ValueError(
                f"LoRA input last dimension ({arr.shape[-1] if arr.ndim > 0 else 'scalar'}) "
                f"does not match in_features ({self.in_features})."
            )
        _assert_axis_dimension(arr, -1, self.in_features)
        a_val = ensure_array(self.lora_a.value)
        b_val = ensure_array(self.lora_b.value)
        if self.promote_dtype is not None:
            promoted = self.promote_dtype(arr, a_val, b_val, dtype=self.dtype)
            arr, a_val, b_val = promoted[0], promoted[1], promoted[2]
        if self.precision is not None:
            last_ax = max(0, arr.ndim - 1)
            hidden: JaxArray = lax.dot_general(
                arr,
                a_val,
                dimension_numbers=(((last_ax,), (0,)), ((), ())),
                precision=self.precision,
            )
            lora_out: JaxArray = lax.dot_general(
                hidden,
                b_val,
                dimension_numbers=(((max(0, hidden.ndim - 1),), (0,)), ((), ())),
                precision=self.precision,
            )
        else:
            hidden = jnp.dot(arr, a_val)
            lora_out = jnp.dot(hidden, b_val)
        if self.base_module is not None and callable(self.base_module):
            base_out: JaxArray = ensure_array(self.base_module(arr))
            return base_out + lora_out
        return lora_out


class LoRALinear(Linear):
    """Linear layer combined with a built-in Low-Rank Adaptation (LoRA) branch.

    Attributes:
        base_module: Optional external base ``Module`` overriding the linear path.
        lora: Child ``LoRA`` adapter module.
    """

    base_module: Optional[Module]
    lora: LoRA

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_module: Optional[Module] = None,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        precision: PrecisionSpec = None,
        *,
        lora_rank: int = 4,
        a_initializer: InitializerLike = default_kernel_init,
        b_initializer: InitializerLike = default_bias_init,
        lora_param_type: type[Variable[JaxArray]] = LoRAParam,
        lora_promote_dtype: PromoteDtypeFn = default_promote_dtype,
        a_metadata: MetadataMapping = EMPTY_METADATA,
        b_metadata: MetadataMapping = EMPTY_METADATA,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        lora_dtype: Optional[DType] = None,
        lora_param_dtype: DType = "float32",
        use_bias: bool = True,
        kernel_init: InitializerLike = default_kernel_init,
        bias_init: InitializerLike = default_bias_init,
        dot_general: DotGeneralLike = None,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        preferred_element_type: Optional[DType] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a LoRALinear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            base_module: Optional base module override.
            dtype: Linear computation data type.
            param_dtype: Linear parameter initialization data type.
            precision: Numerical precision specification.
            lora_rank: Rank of the LoRA adapter matrices. Defaults to ``4``.
            a_initializer: Initializer for LoRA ``lora_a``.
            b_initializer: Initializer for LoRA ``lora_b``.
            lora_param_type: Variable subclass for LoRA parameters.
            lora_promote_dtype: Dtype promotion callable for the LoRA branch.
            a_metadata: Metadata mapping for ``lora_a``.
            b_metadata: Metadata mapping for ``lora_b``.
            rngs: Random number generator stream(s).
            lora_dtype: Optional computation dtype for the LoRA branch.
            lora_param_dtype: Parameter dtype for the LoRA branch.
            use_bias: Whether the linear branch includes a bias vector.
            kernel_init: Initializer for the linear weight matrix.
            bias_init: Initializer for the linear bias vector.
            dot_general: Generalized dot product callable.
            promote_dtype: Dtype promotion callable for the linear branch.
            preferred_element_type: Optional accumulation dtype.
            kernel_metadata: Metadata mapping for ``kernel``.
            bias_metadata: Metadata mapping for ``bias``.
            **kwargs: Additional configuration metadata entries.
        """
        eff_rngs: Union[Rngs, RngStream] = rngs if rngs is not None else Rngs(0)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dot_general=dot_general,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=eff_rngs,
            kernel_metadata=kernel_metadata,
            bias_metadata=bias_metadata,
        )
        for k_name, k_val in kwargs.items():
            self.register_static(k_name, k_val)
        if isinstance(base_module, Module):
            self.base_module = self.register_child("base_module", base_module)
        else:
            self.base_module = None
        self.lora = self.register_child(
            "lora",
            LoRA(
                in_features=in_features,
                lora_rank=lora_rank,
                out_features=out_features,
                precision=precision,
                base_module=None,
                dtype=lora_dtype if lora_dtype is not None else dtype,
                param_dtype=lora_param_dtype,
                a_initializer=a_initializer,
                b_initializer=b_initializer,
                lora_param_type=lora_param_type,
                promote_dtype=lora_promote_dtype,
                rngs=eff_rngs,
                a_metadata=a_metadata,
                b_metadata=b_metadata,
            ),
        )
        self.finish_init()

    def __call__(self, inputs: ArrayLike) -> JaxArray:
        """Compute the sum of the linear projection (or ``base_module``) and the LoRA branch.

        Args:
            inputs: Input tensor with last dimension ``in_features``.

        Returns:
            Output ``JaxArray`` with last dimension ``out_features``.
        """
        if self.base_module is not None and callable(self.base_module):
            linear_out: JaxArray = ensure_array(self.base_module(inputs))
        else:
            linear_out = super().__call__(inputs)
        lora_out = self.lora(inputs)
        return linear_out + lora_out


__all__ = [
    "Dense",
    "Einsum",
    "Linear",
    "LinearGeneral",
    "LoRA",
    "LoRALinear",
    "default_bias_init",
    "default_kernel_init",
]
