from zero_flax.nnx.module import Module
from typing import (
    Any,
    Callable,
    Sequence,
    Optional,
    Iterable,
    Mapping,
    Type,
)
import numpy

Array = Any
ArrayLike = Any


class AxisName:
    pass


class PrecisionLike:
    pass


class PaddingLike:
    pass


class Dtype:
    pass


class Shape:
    pass


class Axes:
    pass


class Size:
    pass


class Axis:
    pass


class DotGeneralT:
    pass


class MaxFun:
    pass


class filterlib:
    class Filter:
        pass


class rnglib:
    class Rngs:
        pass


class variables:
    class Variable:
        pass


class chex:
    class Array:
        pass

    class Numeric:
        pass

    class Scalar:
        pass


class core:
    class Shape:
        pass


class optax:
    class _src:
        class base:
            class GradientTransformationExtraArgs:
                pass


class base:
    class GradientTransformation:
        pass

    class Schedule:
        pass


class jax:
    class Array:
        pass

    class Device:
        pass

    class _src:
        class typing:
            class SupportsDType:
                pass


M = Any
A = Any
UNSPECIFIED = None
_UNSPECIFIED = None
default_kernel_init = None
default_bias_init = None
default_embed_init = None
lax = Any
FrozenDict = Any
KeyArray = Any
RealNumeric = Any
LoRAParam = Any
dot_product_attention = None


class Initializer:
    pass


class initializers:
    zeros = None
    ones = None
    zeros_init = None
    ones_init = None


class BatchNorm(Module):
    """BatchNorm Module."""

    def __init__(
        self,
        num_features: int,
        use_running_average: bool = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-05,
        dtype: Optional[Dtype] = None,
        param_dtype: str | type[Any] | numpy.dtype | Any | Any = None,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Any | Callable[..., Any] = None,
        scale_init: Any | Callable[..., Any] = None,
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: Any = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class ConvTranspose(Module):
    """Convolution Module wrapping ``lax.conv_transpose``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] | None = None,
        padding: PaddingLike = "SAME",
        kernel_dilation: int | Sequence[int] | None = None,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: Dtype | None = None,
        param_dtype=None,
        precision: PrecisionLike | None = None,
        kernel_init=None,
        bias_init=None,
        transpose_kernel: bool = False,
        rngs: Any = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class Dropout(Module):
    """Create a dropout layer."""

    def __init__(
        self,
        rate: float,
        broadcast_dims: Sequence[int] = (),
        deterministic: bool = False,
        rng_collection: str = "dropout",
        rngs: Any | None = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class Jit(Module):
    """Abstract base class for generic types."""

    def __init__(
        self,
        module_constructor: Callable[..., M],
        in_shardings=None,
        out_shardings=None,
        static_argnums: int | Sequence[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        donate_argnums: int | Sequence[int] | None = None,
        donate_argnames: str | Iterable[str] | None = None,
        keep_unused: bool = False,
        device: Optional[Any] = None,
        backend: Optional[str] = None,
        inline: bool = False,
        abstracted_axes: Optional[Any] = None,
        donate_state: bool = False,
        constrain_state: bool | Any = False,
        module_init_args: tuple[Any, ...] = None,
        module_init_kwargs: dict[str, Any] = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class LoRA(Module):
    """A standalone LoRA layer."""

    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        base_module: Optional[Module] = None,
        dtype: Optional[Dtype] = None,
        param_dtype: str | type[Any] | numpy.dtype | Any | Any = None,
        kernel_init: Any | Callable[..., Any] = None,
        lora_param_type: Type[variables.Variable] = None,
        rngs: Any = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class LoRALinear(Module):
    """An `nnx.Linear` layer in which the output will be LoRAified."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_dtype: Optional[Dtype] = None,
        lora_param_dtype: str | type[Any] | numpy.dtype | Any | Any = None,
        lora_kernel_init: Any | Callable[..., Any] = None,
        lora_param_type: Type[variables.Variable] = None,
        rngs: Any = None,
        kwargs={},
        use_bias: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = None,
        precision: PrecisionLike = None,
        kernel_init: Initializer = None,
        bias_init: Initializer = None,
        dot_general: DotGeneralT = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class MultiHeadAttention(Module):
    """Multi-head attention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        dtype: Dtype | None = None,
        param_dtype=None,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: PrecisionLike = None,
        kernel_init=None,
        out_kernel_init: Initializer | None = None,
        bias_init=None,
        out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        attention_fn: Callable[..., Array] = "(dot_product_attention)",
        decode: bool | None = None,
        normalize_qk: bool = False,
        qkv_dot_general: DotGeneralT | None = None,
        out_dot_general: DotGeneralT | None = None,
        qkv_dot_general_cls: Any = None,
        out_dot_general_cls: Any = None,
        rngs: Any = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class Pmap(Module):
    """Abstract base class for generic types."""

    def __init__(
        self,
        module_constructor: Callable[..., M],
        axis_name: AxisName | None = None,
        in_axes: Any = 0,
        out_axes: Any = 0,
        static_broadcasted_argnums: int | Iterable[int] = (),
        devices: Sequence[Any] | None = None,
        backend: str | None = None,
        axis_size: int | None = None,
        donate_argnums: int | Iterable[int] = (),
        global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
        in_axes_kwargs: Any = 0,
        state_axes: Mapping[filterlib.Filter, int] = None,
        split_rngs: filterlib.Filter = Ellipsis,
        transform_metadata: Mapping[str, Any] = None,
        module_init_args: tuple[Any, ...] = None,
        module_init_kwargs: dict[str, Any] = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class Remat(Module):
    """Abstract base class for generic types."""

    def __init__(
        self,
        module_constructor: Callable[..., M],
        prevent_cse: bool = True,
        static_argnums: int | tuple[int, ...] = (),
        policy: Callable[..., bool] | None = None,
        module_init_args: tuple[Any, ...] = None,
        module_init_kwargs: dict[str, Any] = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class Scan(Module):
    """Abstract base class for generic types."""

    def __init__(
        self,
        module_constructor: Callable[..., M],
        length: int | None = None,
        reverse: bool = False,
        unroll: int | bool = 1,
        _split_transpose: bool = False,
        in_axes: int | None | Sequence[Any] = 0,
        in_axes_kwargs: Any = 0,
        out_axes: Any = 0,
        carry_argnum: int = 1,
        state_axes: Mapping[filterlib.Filter, int] = None,
        split_rngs: filterlib.Filter = Ellipsis,
        transform_metadata: Mapping[str, Any] = None,
        scan_output: bool = True,
        module_init_args: tuple[Any, ...] = None,
        module_init_kwargs: dict[str, Any] = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class Vmap(Module):
    """Abstract base class for generic types."""

    def __init__(
        self,
        module_constructor: Callable[..., M],
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
        axis_name: AxisName | None = None,
        axis_size: int | None = None,
        spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
        in_axes_kwargs: Any = 0,
        state_axes: Mapping[filterlib.Filter, int] = None,
        split_rngs: filterlib.Filter = Ellipsis,
        transform_metadata: Mapping[str, Any] = None,
        module_init_args: tuple[Any, ...] = None,
        module_init_kwargs: dict[str, Any] = None,
    ):
        super().__init__()
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass
