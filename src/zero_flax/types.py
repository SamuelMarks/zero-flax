"""Strong domain type system and helper utilities for zero-flax.

This module defines concrete protocols, structural types, runtime-instantiable
type classes (`Dtype`, `Shape`, `Axis`, `Axes`, `Size`, `AxisName`, `PaddingLike`,
`PrecisionLike`, `Initializer`, `DotGeneralT`, `MaxFun`), and tensor helper routines
without using `typing.Any`, `object`, `typing.cast`, or dynamic reflection.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_ir.types import DType as IrDType
from typing_extensions import TypeAlias, TypeGuard
from zero_jax import lax
from zero_jax import nn as jnn
from zero_jax import numpy as jnp
from zero_jax.nn import initializers as jnn_initializers
from zero_jax.numpy.lax_numpy import ndarray as JaxArray

Scalar: TypeAlias = Union[int, float, bool, complex]
NestedScalarSequence: TypeAlias = Sequence[Union[Scalar, "NestedScalarSequence"]]

VALID_DTYPE_STRINGS: frozenset[str] = frozenset(
    {
        "float32",
        "float64",
        "float16",
        "bfloat16",
        "int32",
        "int64",
        "int16",
        "int8",
        "uint32",
        "uint64",
        "uint16",
        "uint8",
        "bool",
        "complex64",
        "complex128",
    }
)


@runtime_checkable
class ArrayProtocol(Protocol):
    """Structural protocol for array and tensor objects.

    Attributes:
        shape: Tuple of integer dimension sizes.
        ndim: Number of array dimensions.
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the array as a tuple of integers.

        Returns:
            Tuple of dimension sizes.
        """

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array.

        Returns:
            Number of dimensions.
        """


Array: TypeAlias = Union[JaxArray, Tensor, ArrayProtocol]
ArrayLike: TypeAlias = Union[Array, Scalar, NestedScalarSequence]


class Dtype:
    """Representation of a tensor data type supporting both type annotations and runtime instantiation.

    Attributes:
        name: Canonical string name of the data type.
    """

    name: str

    def __init__(self, name: str = "float32") -> None:
        """Initialize a Dtype descriptor.

        Args:
            name: Canonical string representation of the dtype (e.g., ``"float32"``).
        """
        self.name = str(name)

    def __repr__(self) -> str:
        """Return a string representation of the Dtype.

        Returns:
            Formatted string representation.
        """
        return f"Dtype({self.name!r})"

    def __str__(self) -> str:
        """Return the canonical dtype name.

        Returns:
            Canonical string name.
        """
        return self.name


DType: TypeAlias = Union[
    Dtype,
    str,
    IrDType,
    CompilerDType,
    type[int],
    type[float],
    type[bool],
    type[str],
]


class Shape:
    """Descriptor representing a tensor shape that can be instantiated at runtime and iterated.

    Attributes:
        dims: Tuple of integer dimensions.
    """

    dims: Tuple[int, ...]

    def __init__(self, dims: Sequence[int] = ()) -> None:
        """Initialize a Shape descriptor.

        Args:
            dims: Sequence of integer dimension sizes.
        """
        self.dims = tuple(int(d) for d in dims)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the integer dimensions of the shape.

        Returns:
            Iterator over ``self.dims``.
        """
        return iter(self.dims)

    def __len__(self) -> int:
        """Return the rank (number of dimensions) of the shape.

        Returns:
            Length of ``self.dims``.
        """
        return len(self.dims)


class Axis:
    """Descriptor representing a single tensor axis index.

    Attributes:
        index: Integer axis index.
    """

    index: int

    def __init__(self, index: int = 0) -> None:
        """Initialize an Axis descriptor.

        Args:
            index: Axis index value.
        """
        self.index = int(index)

    def __int__(self) -> int:
        """Convert the Axis to a plain integer.

        Returns:
            Integer axis index.
        """
        return self.index


class Axes:
    """Representation of one or more tensor axes supporting runtime instantiation.

    Attributes:
        axes: Normalized tuple of integer axis indices.
    """

    axes: Tuple[int, ...]

    def __init__(self, axes: Union[int, Sequence[int]] = (-1,)) -> None:
        """Initialize an Axes specification.

        Args:
            axes: Single integer axis or sequence of integer axes.
        """
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = tuple(int(a) for a in axes)

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over the underlying tuple of axes.

        Returns:
            Iterator over integer axes.
        """
        return iter(self.axes)


class Size:
    """Descriptor representing a dimension or feature size.

    Attributes:
        value: Integer size value.
    """

    value: int

    def __init__(self, value: int = 1) -> None:
        """Initialize a Size descriptor.

        Args:
            value: Integer dimension size.
        """
        self.value = int(value)

    def __int__(self) -> int:
        """Convert the Size to a plain integer.

        Returns:
            Integer size value.
        """
        return self.value


class AxisName:
    """Descriptor representing a named parallel or logical axis.

    Attributes:
        label: Axis name string.
    """

    label: str

    def __init__(self, label: str = "batch") -> None:
        """Initialize an AxisName descriptor.

        Args:
            label: Name of the axis.
        """
        self.label = str(label)

    def __str__(self) -> str:
        """Return the string label of the axis.

        Returns:
            Axis label string.
        """
        return self.label


class PaddingLike:
    """Padding specification descriptor supporting string modes and explicit padding tuples.

    Attributes:
        padding: Padding mode string or sequence of padding pairs.
    """

    padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]]

    def __init__(
        self,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = "SAME",
    ) -> None:
        """Initialize a PaddingLike specification.

        Args:
            padding: Padding mode (``"SAME"``, ``"VALID"``, ``"CAUSAL"``, ``"CIRCULAR"``)
                or sequence of integer padding amounts.
        """
        self.padding = padding


PaddingSpec: TypeAlias = Union[PaddingLike, str, int, Sequence[int], Sequence[Tuple[int, int]]]


class PrecisionLike:
    """Numerical precision specification for matrix multiplication and convolution.

    Attributes:
        precision: Precision mode string or tuple of precision mode strings.
    """

    precision: Union[str, Tuple[str, str], None]

    def __init__(self, precision: Union[str, Tuple[str, str], None] = None) -> None:
        """Initialize a PrecisionLike descriptor.

        Args:
            precision: Optional precision level name or pair of names.
        """
        self.precision = precision


PrecisionSpec: TypeAlias = Union[PrecisionLike, str, Tuple[str, str], None]

AxisIndexGroups: TypeAlias = Optional[Sequence[Sequence[int]]]
MetadataValue: TypeAlias = Union[
    str, int, float, bool, None, Tuple[int, ...], Sequence[Optional[str]]
]
MetadataMapping: TypeAlias = Optional[Mapping[str, MetadataValue]]
EMPTY_METADATA: Mapping[str, MetadataValue] = MappingProxyType({})

PRNGKeyLike: TypeAlias = Union[JaxArray, Tensor, ArrayProtocol, Tuple[int, int], int, None]


@runtime_checkable
class InitializerProtocol(Protocol):
    """Protocol for parameter initialization callables."""

    def __call__(
        self,
        key: PRNGKeyLike,
        shape: Sequence[int],
        dtype: Optional[DType] = ...,
    ) -> JaxArray:
        """Generate an initialized parameter array.

        Args:
            key: PRNG key for stochastic initialization.
            shape: Target shape of the parameter tensor.
            dtype: Desired parameter data type.

        Returns:
            Initialized ``JaxArray`` of shape ``shape``.
        """


@runtime_checkable
class TwoArgInitializerProtocol(Protocol):
    """Protocol for 2-argument parameter initialization callables ``(key, shape) -> JaxArray``."""

    def __call__(
        self,
        key: PRNGKeyLike,
        shape: Sequence[int],
    ) -> JaxArray:
        """Generate an initialized parameter array from ``(key, shape)``.

        Args:
            key: PRNG key for stochastic initialization.
            shape: Target shape of the parameter tensor.

        Returns:
            Initialized ``JaxArray`` of shape ``shape``.
        """


InitializerCallable: TypeAlias = Union[
    InitializerProtocol,
    TwoArgInitializerProtocol,
    Callable[[PRNGKeyLike, Sequence[int], Optional[DType]], JaxArray],
    Callable[[PRNGKeyLike, Sequence[int]], JaxArray],
]


class Initializer:
    """Concrete callable wrapper for parameter initializers.

    Attributes:
        fn: Underlying initializer callable.
    """

    fn: Callable[[PRNGKeyLike, Sequence[int], Optional[DType]], JaxArray]

    def __init__(
        self,
        fn: Optional[Callable[[PRNGKeyLike, Sequence[int], Optional[DType]], JaxArray]] = None,
    ) -> None:
        """Initialize an Initializer wrapper.

        Args:
            fn: Optional custom initializer callable. Defaults to ``zeros``.
        """
        self.fn = fn if fn is not None else default_bias_init

    def __call__(
        self,
        key: PRNGKeyLike,
        shape: Sequence[int],
        dtype: Optional[DType] = "float32",
    ) -> JaxArray:
        """Invoke the initializer function to produce a parameter tensor.

        Args:
            key: PRNG key for random generation.
            shape: Desired parameter tensor shape.
            dtype: Desired parameter tensor dtype.

        Returns:
            Initialized ``JaxArray`` instance.
        """
        return self.fn(key, shape, dtype)


InitializerLike: TypeAlias = Union[
    InitializerCallable,
    Initializer,
    None,
]

ActivationFn: TypeAlias = Union[Callable[[ArrayLike], JaxArray], str]


@runtime_checkable
class PromoteDtypeProtocol(Protocol):
    """Protocol for dtype promotion callables."""

    def __call__(
        self,
        *args: Optional[ArrayLike],
        dtype: Optional[DType] = ...,
        inexact: bool = ...,
    ) -> list[JaxArray]:
        """Promote input arrays to a common dtype.

        Args:
            *args: Input arrays or ``None``.
            dtype: Target dtype override.
            inexact: Whether integer types are promoted to float.

        Returns:
            List of promoted ``JaxArray`` tensors.
        """


PromoteDtypeFn: TypeAlias = Optional[PromoteDtypeProtocol]

DimensionNumbers: TypeAlias = Tuple[
    Tuple[Sequence[int], Sequence[int]],
    Tuple[Sequence[int], Sequence[int]],
]


@runtime_checkable
class DotGeneralProtocol(Protocol):
    """Protocol for generalized matrix contraction callables."""

    def __call__(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        dimension_numbers: DimensionNumbers,
        precision: PrecisionSpec = ...,
        preferred_element_type: Optional[DType] = ...,
    ) -> JaxArray:
        """Contract ``lhs`` and ``rhs`` according to ``dimension_numbers``.

        Args:
            lhs: Left operand array.
            rhs: Right operand array.
            dimension_numbers: Contracting and batch dimension pairs.
            precision: Optional precision specification.
            preferred_element_type: Optional accumulation dtype.

        Returns:
            Contracted ``JaxArray``.
        """


class DotGeneralT:
    """Callable wrapper for generalized dot product contractions.

    Attributes:
        fn: Underlying generalized contraction callable.
    """

    fn: Optional[DotGeneralProtocol]

    def __init__(self, fn: Optional[DotGeneralProtocol] = None) -> None:
        """Initialize a DotGeneralT callable wrapper.

        Args:
            fn: Optional custom ``dot_general`` callable. Defaults to ``lax.dot_general``.
        """
        self.fn = fn

    def __call__(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        dimension_numbers: DimensionNumbers,
        precision: PrecisionSpec = None,
        preferred_element_type: Optional[DType] = None,
    ) -> JaxArray:
        """Compute generalized contraction between ``lhs`` and ``rhs``.

        Args:
            lhs: Left-hand side array operand.
            rhs: Right-hand side array operand.
            dimension_numbers: Tuple of ``((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))``.
            precision: Optional numerical precision specification.
            preferred_element_type: Optional output accumulation dtype.

        Returns:
            Contracted result as a ``JaxArray``.
        """
        if self.fn is not None:
            return self.fn(
                lhs,
                rhs,
                dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
        contract_dims, batch_dims = dimension_numbers
        lhs_arr = ensure_array(lhs)
        rhs_arr = ensure_array(rhs)
        if len(batch_dims[0]) == 0 and len(batch_dims[1]) == 0:
            res: JaxArray = jnp.tensordot(lhs_arr, rhs_arr, axes=contract_dims)
        else:
            res = lax.dot_general(
                lhs_arr,
                rhs_arr,
                dimension_numbers=dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
        if preferred_element_type is not None:
            res = ensure_array(res, dtype=preferred_element_type)
        return res


DotGeneralLike: TypeAlias = Union[DotGeneralT, DotGeneralProtocol, None]


@runtime_checkable
class ConvGeneralDilatedProtocol(Protocol):
    """Protocol for general dilated convolution callables."""

    def __call__(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
    ) -> JaxArray:
        """Compute dilated convolution over ``(lhs, rhs)``.

        Args:
            lhs: Input activation tensor.
            rhs: Convolution kernel tensor.

        Returns:
            Convolved ``JaxArray``.
        """


ConvGeneralDilatedT: TypeAlias = Optional[ConvGeneralDilatedProtocol]


@runtime_checkable
class EinsumProtocol(Protocol):
    """Protocol for Einstein summation callables."""

    def __call__(
        self,
        subscripts: str,
        lhs: ArrayLike,
        rhs: ArrayLike,
    ) -> JaxArray:
        """Evaluate Einstein summation over ``(lhs, rhs)``.

        Args:
            subscripts: Einstein summation subscript specification.
            lhs: Left operand array.
            rhs: Right operand array.

        Returns:
            Contracted ``JaxArray``.
        """


EinsumOpT: TypeAlias = Optional[EinsumProtocol]


@runtime_checkable
class AttentionFnProtocol(Protocol):
    """Protocol for scaled dot-product attention callables."""

    def __call__(
        self,
        query: JaxArray,
        key: JaxArray,
        value: JaxArray,
        *,
        mask: Optional[ArrayLike] = ...,
        dropout_rng: PRNGKeyLike = ...,
        dropout_rate: float = ...,
        broadcast_dropout: bool = ...,
        deterministic: bool = ...,
        dtype: Optional[DType] = ...,
        precision: PrecisionSpec = ...,
    ) -> JaxArray:
        """Compute attention over ``(query, key, value)``.

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
            Attended ``JaxArray``.
        """


AttentionFnT: TypeAlias = Optional[AttentionFnProtocol]


class MaxFun:
    """Callable wrapper computing the maximum of two arrays or along an axis.

    Attributes:
        axis: Optional default reduction axis.
    """

    axis: Optional[int]

    def __init__(self, axis: Optional[int] = None) -> None:
        """Initialize a MaxFun descriptor.

        Args:
            axis: Optional default reduction axis.
        """
        self.axis = axis

    def __call__(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> JaxArray:
        """Compute elementwise maximum if ``y`` is provided, else reduction maximum.

        Args:
            x: First input array.
            y: Optional second input array for elementwise comparison.

        Returns:
            Maximum result as a ``JaxArray``.
        """
        x_arr = ensure_array(x)
        if y is not None:
            res_elem: JaxArray = jnp.maximum(x_arr, ensure_array(y))
            return res_elem
        res_red: JaxArray = jnp.max(x_arr, axis=self.axis)
        return res_red


def is_tensor(value: Optional[ArrayLike]) -> TypeGuard[Tensor]:
    """Check whether ``value`` is an ``ml_switcheroo_compiler.core.tensor.Tensor``.

    Args:
        value: Candidate value to inspect.

    Returns:
        ``True`` if ``value`` is a ``Tensor`` instance, otherwise ``False``.
    """
    return isinstance(value, Tensor)


def is_int_sequence(value: Union[int, Sequence[int]]) -> TypeGuard[Sequence[int]]:
    """Check whether ``value`` is a non-integer sequence of integers.

    Args:
        value: Single integer or sequence of integers.

    Returns:
        ``True`` if ``value`` is a sequence of integers, ``False`` if it is an ``int``.
    """
    return not isinstance(value, int)


def is_padding_pair_sequence(
    value: PaddingSpec,
) -> TypeGuard[Sequence[Tuple[int, int]]]:
    """Check whether ``value`` is a sequence of ``(low, high)`` integer padding pairs.

    Args:
        value: Padding specification to inspect.

    Returns:
        ``True`` if ``value`` is a sequence of 2-tuples of ints.
    """
    if isinstance(value, (str, int, PaddingLike)):
        return False
    if len(value) == 0:
        return False
    first = value[0]
    return isinstance(first, tuple) and len(first) == 2


def normalize_int_tuple(
    value: Union[int, Sequence[int], None], length: int, default: int = 1
) -> Tuple[int, ...]:
    """Normalize an integer or sequence of integers into a tuple of a given length.

    Args:
        value: Single integer, sequence of integers, or ``None``.
        length: Target tuple length.
        default: Default integer value when ``value`` is ``None`` or empty string.

    Returns:
        Tuple of ``length`` integers.
    """
    if value is None:
        return (default,) * length
    if isinstance(value, int):
        return (value,) * length
    seq = tuple(int(v) for v in value)
    if len(seq) == 1 and length > 1:
        return seq * length
    return seq


def sanitize_dtype_str(dtype: Optional[DType], default: str = "float32") -> str:
    """Normalize a ``DType`` descriptor into a canonical dtype string.

    Args:
        dtype: Candidate dtype descriptor, enum, Python scalar type, or ``None``.
        default: Fallback dtype string when ``dtype`` is ``None``.

    Returns:
        Normalized dtype string (e.g., ``"float32"``).

    Raises:
        ValueError: If ``dtype`` is an unrecognized dtype string.
    """
    if dtype is None:
        return default
    if isinstance(dtype, Dtype):
        if dtype.name in VALID_DTYPE_STRINGS:
            return dtype.name
        raise ValueError(f"Unsupported Dtype name: {dtype.name!r}")
    if isinstance(dtype, (IrDType, CompilerDType)):
        return str(dtype.value).lower()
    if dtype is bool:
        return "bool"
    if dtype is int:
        return "int32"
    if dtype is float:
        return "float32"
    if isinstance(dtype, str):
        cleaned_str = dtype.lower().removeprefix("dtype.")
        if cleaned_str in VALID_DTYPE_STRINGS:
            return cleaned_str
    raise ValueError(f"Unsupported dtype string: {dtype!r}")


def ensure_array(value: ArrayLike, dtype: Optional[DType] = None) -> JaxArray:
    """Convert any ``ArrayLike`` input into a ``zero_jax`` ``ndarray``.

    Args:
        value: Input tensor, array, scalar, or nested sequence.
        dtype: Optional target dtype.

    Returns:
        Converted ``JaxArray``.
    """
    if isinstance(value, JaxArray) and dtype is None:
        return value
    if isinstance(value, ArrayProtocol) and not isinstance(value, JaxArray) and 0 in value.shape:
        clean_zero_dt = sanitize_dtype_str(dtype)
        zero_res: JaxArray = jnp.zeros(tuple(int(d) for d in value.shape), dtype=clean_zero_dt)
        return zero_res
    if dtype is not None:
        clean_dt = sanitize_dtype_str(dtype)
        res_typed: JaxArray = jnp.asarray(value, dtype=clean_dt)
        return res_typed
    res_untyped: JaxArray = jnp.asarray(value)
    return res_untyped


_FLOAT_DTYPE_PRIORITY: dict[str, int] = {
    "float16": 1,
    "bfloat16": 2,
    "float32": 3,
    "float64": 4,
    "complex64": 5,
    "complex128": 6,
}


def default_promote_dtype(
    *args: Optional[ArrayLike],
    dtype: Optional[DType] = None,
    inexact: bool = True,
) -> list[JaxArray]:
    """Promote input array operands to a common data type.

    Args:
        *args: Input arrays or ``None`` entries.
        dtype: Optional target dtype override.
        inexact: Whether to promote integer/boolean types to floating-point.

    Returns:
        List of converted ``JaxArray`` operands (with ``zeros(0)`` placeholder for ``None`` entries).
    """
    resolved_dt: Optional[str] = sanitize_dtype_str(dtype) if dtype is not None else None
    converted: list[Optional[JaxArray]] = []
    needs_float = False
    seen_float_dts: list[str] = []
    for item in args:
        if item is None:
            converted.append(None)
        else:
            arr = ensure_array(item, dtype=resolved_dt)
            dt_str = sanitize_dtype_str(arr.dtype)
            if inexact and resolved_dt is None and ("int" in dt_str or "bool" in dt_str):
                needs_float = True
            elif dt_str in _FLOAT_DTYPE_PRIORITY:
                seen_float_dts.append(dt_str)
            converted.append(arr)

    target_dt: Optional[str] = resolved_dt
    if target_dt is None:
        if len(set(seen_float_dts)) > 1:
            target_dt = max(seen_float_dts, key=lambda d: _FLOAT_DTYPE_PRIORITY.get(d, 3))
        elif needs_float:
            target_dt = (
                max(seen_float_dts, key=lambda d: _FLOAT_DTYPE_PRIORITY.get(d, 3))
                if seen_float_dts
                else "float32"
            )

    out: list[JaxArray] = []
    for arr_item in converted:
        if arr_item is None:
            out.append(jnp.zeros((0,), dtype=target_dt if target_dt is not None else "float32"))
        elif target_dt is not None:
            out.append(ensure_array(arr_item, dtype=target_dt))
        else:
            out.append(arr_item)
    return out


def default_kernel_init(
    key: PRNGKeyLike,
    shape: Sequence[int],
    dtype: Optional[DType] = "float32",
) -> JaxArray:
    """Default LeCun normal kernel initializer used by Flax NNX layers.

    Args:
        key: PRNG key or seed.
        shape: Shape of the weight tensor to initialize.
        dtype: Desired weight tensor dtype.

    Returns:
        Initialized weight ``JaxArray``.
    """
    eff_key: PRNGKeyLike = (0, 0) if key is None else key
    init_fn = jnn_initializers.lecun_normal()
    res: JaxArray = init_fn(eff_key, tuple(shape), sanitize_dtype_str(dtype))
    return res


def default_bias_init(
    key: PRNGKeyLike,
    shape: Sequence[int],
    dtype: Optional[DType] = "float32",
) -> JaxArray:
    """Default zero bias initializer used by Flax NNX layers.

    Args:
        key: PRNG key or seed.
        shape: Shape of the bias tensor to initialize.
        dtype: Desired bias tensor dtype.

    Returns:
        Zero-initialized ``JaxArray``.
    """
    eff_key: PRNGKeyLike = (0, 0) if key is None else key
    res: JaxArray = jnn_initializers.zeros(eff_key, tuple(shape), sanitize_dtype_str(dtype))
    return res


def ones_init(
    key: PRNGKeyLike,
    shape: Sequence[int],
    dtype: Optional[DType] = "float32",
) -> JaxArray:
    """Ones initializer for normalization scale parameters.

    Args:
        key: PRNG key or seed.
        shape: Shape of the scale tensor to initialize.
        dtype: Desired scale tensor dtype.

    Returns:
        Ones-initialized ``JaxArray``.
    """
    eff_key: PRNGKeyLike = (0, 0) if key is None else key
    res: JaxArray = jnn_initializers.ones(eff_key, tuple(shape), sanitize_dtype_str(dtype))
    return res


def zeros_init(
    key: PRNGKeyLike,
    shape: Sequence[int],
    dtype: Optional[DType] = "float32",
) -> JaxArray:
    """Zeros initializer for bias and carry parameters.

    Args:
        key: PRNG key or seed.
        shape: Shape of the tensor to initialize.
        dtype: Desired tensor dtype.

    Returns:
        Zero-initialized ``JaxArray``.
    """
    eff_key: PRNGKeyLike = (0, 0) if key is None else key
    res: JaxArray = jnn_initializers.zeros(eff_key, tuple(shape), sanitize_dtype_str(dtype))
    return res


def default_embed_init(
    key: PRNGKeyLike,
    shape: Sequence[int],
    dtype: Optional[DType] = "float32",
) -> JaxArray:
    """Default variance-scaling normal initializer for embedding tables.

    Args:
        key: PRNG key or seed.
        shape: Shape ``(num_embeddings, features)`` of the embedding table.
        dtype: Desired embedding dtype.

    Returns:
        Initialized embedding ``JaxArray``.
    """
    eff_key: PRNGKeyLike = (0, 0) if key is None else key
    init_fn = jnn_initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    res: JaxArray = init_fn(eff_key, tuple(shape), sanitize_dtype_str(dtype))
    return res


def orthogonal_init(
    key: PRNGKeyLike,
    shape: Sequence[int],
    dtype: Optional[DType] = "float32",
) -> JaxArray:
    """Orthogonal matrix initializer for recurrent weights.

    Args:
        key: PRNG key or seed.
        shape: Shape of the recurrent kernel tensor.
        dtype: Desired tensor dtype.

    Returns:
        Initialized recurrent weight ``JaxArray``.
    """
    eff_key: PRNGKeyLike = (0, 0) if key is None else key
    init_fn = jnn_initializers.orthogonal()
    res: JaxArray = init_fn(eff_key, tuple(shape), sanitize_dtype_str(dtype))
    return res


def resolve_initializer(
    init: InitializerLike,
    fallback: Callable[
        [PRNGKeyLike, Sequence[int], Optional[DType]], JaxArray
    ] = default_kernel_init,
) -> Callable[[PRNGKeyLike, Sequence[int], Optional[DType]], JaxArray]:
    """Resolve an ``InitializerLike`` argument into a 3-argument initializer callable.

    Args:
        init: User-supplied initializer callable, ``Initializer`` wrapper, or ``None``.
        fallback: Fallback initializer function when ``init`` is ``None``.

    Returns:
        Callable initializer accepting ``(key, shape, dtype)``.
    """
    if init is None:
        return fallback
    if isinstance(init, Initializer):
        return init

    sig_params = len(inspect.signature(init).parameters)

    def _wrapped(
        key: PRNGKeyLike,
        shape: Sequence[int],
        dtype: Optional[DType] = "float32",
    ) -> JaxArray:
        """Invoke the user-provided initializer with normalized shape and dtype.

        Args:
            key: PRNG key or seed.
            shape: Target tensor shape.
            dtype: Target tensor dtype.

        Returns:
            Initialized ``JaxArray``.
        """
        clean_dt = sanitize_dtype_str(dtype)
        if sig_params == 2 and isinstance(init, TwoArgInitializerProtocol):
            out = init(key, tuple(shape))
        else:
            out = (
                init(key, tuple(shape), clean_dt)
                if isinstance(init, InitializerProtocol)
                else fallback(key, tuple(shape), clean_dt)
            )
        return ensure_array(out)

    return _wrapped


def resolve_activation(fn: ActivationFn) -> Callable[[ArrayLike], JaxArray]:
    """Resolve an activation function or canonical activation name into a callable.

    Args:
        fn: Activation callable or canonical string name (e.g., ``"tanh"``, ``"sigmoid"``, ``"relu"``).

    Returns:
        Callable mapping an ``ArrayLike`` input to a ``JaxArray``.

    Raises:
        ValueError: If ``fn`` is an unrecognized activation string.
    """
    if callable(fn):

        def _call_act(x: ArrayLike) -> JaxArray:
            """Apply the user-provided activation callable to ``x``.

            Args:
                x: Input array.

            Returns:
                Activated ``JaxArray``.
            """
            return ensure_array(fn(ensure_array(x)))

        return _call_act

    lower = fn.strip().lower()
    if lower == "sigmoid":
        return sigmoid_fn
    if lower == "tanh":
        return tanh_fn
    if lower == "relu":

        def _relu(x: ArrayLike) -> JaxArray:
            """Compute rectified linear unit activation ``max(0, x)``.

            Args:
                x: Input array.

            Returns:
                ReLU-activated ``JaxArray``.
            """
            arr = ensure_array(x)
            res: JaxArray = jnp.maximum(arr, 0.0)
            return res

        return _relu
    if lower == "gelu":
        return lambda x: ensure_array(jnn.gelu(ensure_array(x)))
    if lower in ("silu", "swish"):
        return lambda x: ensure_array(jnn.silu(ensure_array(x)))
    if lower == "elu":
        return lambda x: ensure_array(jnn.elu(ensure_array(x)))
    if lower == "celu":
        return lambda x: ensure_array(jnn.celu(ensure_array(x)))
    if lower == "selu":
        return lambda x: ensure_array(jnn.selu(ensure_array(x)))
    if lower == "softplus":
        return lambda x: ensure_array(jnn.softplus(ensure_array(x)))
    if lower == "soft_sign":
        return lambda x: ensure_array(jnn.soft_sign(ensure_array(x)))
    if lower == "hard_tanh":
        return lambda x: ensure_array(jnn.hard_tanh(ensure_array(x)))
    if lower == "hard_sigmoid":
        return lambda x: ensure_array(jnn.hard_sigmoid(ensure_array(x)))
    if lower in ("hard_swish", "hard_silu"):
        return lambda x: ensure_array(ensure_array(x) * jnn.relu6(ensure_array(x) + 3.0) / 6.0)
    if lower == "leaky_relu":
        return lambda x: ensure_array(jnn.leaky_relu(ensure_array(x)))
    if lower == "log_sigmoid":
        return lambda x: ensure_array(jnn.log_sigmoid(ensure_array(x)))
    if lower == "relu6":
        return lambda x: ensure_array(jnn.relu6(ensure_array(x)))
    raise ValueError(f"Unsupported activation function name: {fn!r}")


def sigmoid_fn(x: ArrayLike) -> JaxArray:
    """Compute the elementwise logistic sigmoid ``1 / (1 + exp(-x))``.

    Args:
        x: Input tensor or array.

    Returns:
        Elementwise sigmoid ``JaxArray``.
    """
    arr = ensure_array(x)
    res: JaxArray = 1.0 / (1.0 + jnp.exp(-arr))
    return res


def tanh_fn(x: ArrayLike) -> JaxArray:
    """Compute the elementwise hyperbolic tangent ``tanh(x)``.

    Args:
        x: Input tensor or array.

    Returns:
        Elementwise tanh ``JaxArray``.
    """
    arr = ensure_array(x)
    res: JaxArray = jnp.tanh(arr)
    return res


def compute_dot_general(
    lhs: ArrayLike,
    rhs: ArrayLike,
    contract_axes: Tuple[Sequence[int], Sequence[int]],
) -> JaxArray:
    """Contract ``lhs`` and ``rhs`` along the specified axes using ``jnp.tensordot``.

    Args:
        lhs: Left-hand side array operand.
        rhs: Right-hand side array operand.
        contract_axes: Pair of axis sequences ``(lhs_axes, rhs_axes)`` to contract.

    Returns:
        Contracted ``JaxArray``.
    """
    res: JaxArray = jnp.tensordot(ensure_array(lhs), ensure_array(rhs), axes=contract_axes)
    return res


@runtime_checkable
class GrainRecordProtocol(Protocol):
    """Structural protocol matching ``zero_grain.Record`` instances."""

    @property
    def data(self) -> Union[ArrayLike, Mapping[str, ArrayLike]]:
        """Return the underlying record payload.

        Returns:
            Record ``data`` payload.
        """
        ...


BatchInput: TypeAlias = Union[
    ArrayLike,
    Mapping[str, ArrayLike],
    GrainRecordProtocol,
]


class GrainBatchAdapter:
    """Typed adapter converting ``zero_grain`` ``Record``s, iterators, and batched ``ElementTree``s into ``JaxArray``s.

    Attributes:
        feature_key: Default dictionary key used when extracting a single feature tensor from a mapping batch.
        target_key: Default dictionary key used when extracting a target tensor from a mapping batch.
        dtype: Optional target ``DType`` for converted ``JaxArray`` tensors.
    """

    feature_key: str
    target_key: str
    dtype: Optional[DType]

    def __init__(
        self,
        feature_key: str = "x",
        target_key: str = "y",
        dtype: Optional[DType] = None,
    ) -> None:
        """Initialize a ``GrainBatchAdapter``.

        Args:
            feature_key: Key name for input features in mapping batches. Defaults to ``"x"``.
            target_key: Key name for supervision targets in mapping batches. Defaults to ``"y"``.
            dtype: Optional target ``DType`` for extracted feature tensors.
        """
        self.feature_key = feature_key
        self.target_key = target_key
        self.dtype = dtype

    def to_array(self, batch: BatchInput) -> JaxArray:
        """Extract and convert ``batch`` into a single ``JaxArray``.

        Unwraps ``zero_grain.Record`` via ``.data`` and extracts ``self.feature_key``
        (or the first mapping value) when ``batch`` is a mapping.

        Args:
            batch: ``ArrayLike``, mapping of arrays, or ``zero_grain.Record``.

        Returns:
            Converted ``JaxArray``.
        """
        payload: Union[ArrayLike, Mapping[str, ArrayLike]] = (
            batch.data if isinstance(batch, GrainRecordProtocol) else batch
        )
        if isinstance(payload, Mapping):
            if self.feature_key in payload:
                return ensure_array(payload[self.feature_key], dtype=self.dtype)
            first_val = next(iter(payload.values()))
            return ensure_array(first_val, dtype=self.dtype)
        return ensure_array(payload, dtype=self.dtype)

    def to_feature_target_pair(
        self,
        batch: BatchInput,
    ) -> Tuple[JaxArray, JaxArray]:
        """Extract ``(features, targets)`` as a pair of ``JaxArray``s from a ``zero_grain`` batch or ``Record``.

        Args:
            batch: Mapping or ``zero_grain.Record`` containing ``self.feature_key`` and ``self.target_key``.

        Returns:
            Pair ``(x_array, y_array)`` of ``JaxArray`` tensors.

        Raises:
            KeyError: If ``batch`` does not contain ``self.feature_key`` or ``self.target_key``.
        """
        payload = batch.data if isinstance(batch, GrainRecordProtocol) else batch
        if not isinstance(payload, Mapping):
            raise KeyError("GrainBatchAdapter.to_feature_target_pair requires a mapping batch.")
        x_arr = ensure_array(payload[self.feature_key], dtype=self.dtype)
        y_arr = ensure_array(payload[self.target_key], dtype=self.dtype)
        return x_arr, y_arr

    def to_array_dict(
        self,
        batch: BatchInput,
    ) -> dict[str, JaxArray]:
        """Convert all entries of a ``zero_grain`` mapping batch or ``Record`` into ``JaxArray``s.

        Args:
            batch: Mapping or ``zero_grain.Record`` whose payload is a mapping.

        Returns:
            Dictionary mapping string keys to ``JaxArray`` values.
        """
        payload = batch.data if isinstance(batch, GrainRecordProtocol) else batch
        if not isinstance(payload, Mapping):
            return {self.feature_key: ensure_array(payload, dtype=self.dtype)}
        return {str(k): ensure_array(v, dtype=self.dtype) for k, v in payload.items()}


__all__ = [
    "ActivationFn",
    "Array",
    "ArrayLike",
    "ArrayProtocol",
    "AttentionFnProtocol",
    "AttentionFnT",
    "Axes",
    "Axis",
    "AxisIndexGroups",
    "AxisName",
    "BatchInput",
    "ConvGeneralDilatedProtocol",
    "ConvGeneralDilatedT",
    "DType",
    "DimensionNumbers",
    "DotGeneralLike",
    "DotGeneralProtocol",
    "DotGeneralT",
    "Dtype",
    "EMPTY_METADATA",
    "EinsumOpT",
    "EinsumProtocol",
    "GrainBatchAdapter",
    "GrainRecordProtocol",
    "Initializer",
    "InitializerLike",
    "InitializerProtocol",
    "JaxArray",
    "MaxFun",
    "MetadataMapping",
    "MetadataValue",
    "NestedScalarSequence",
    "PRNGKeyLike",
    "PaddingLike",
    "PaddingSpec",
    "PrecisionLike",
    "PrecisionSpec",
    "PromoteDtypeFn",
    "PromoteDtypeProtocol",
    "Scalar",
    "Shape",
    "Size",
    "TwoArgInitializerProtocol",
    "VALID_DTYPE_STRINGS",
    "compute_dot_general",
    "default_bias_init",
    "default_embed_init",
    "default_kernel_init",
    "default_promote_dtype",
    "ensure_array",
    "is_int_sequence",
    "is_padding_pair_sequence",
    "is_tensor",
    "lax",
    "normalize_int_tuple",
    "ones_init",
    "orthogonal_init",
    "resolve_activation",
    "resolve_initializer",
    "sanitize_dtype_str",
    "sigmoid_fn",
    "tanh_fn",
    "zeros_init",
]
