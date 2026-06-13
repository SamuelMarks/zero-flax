"""Flax (NNX) frontend compatibility layer."""

from typing import Callable, Optional, Union
from collections.abc import Iterable, Sequence


class Module:
    """Base class for all neural network modules in NNX."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the module."""
        for k, v in kwargs.items():
            setattr(self, k, v)


class GraphDef:
    """Represents the static structure of a Module graph."""

    pass


class State(dict):
    """A nested dictionary structure for nnx variables."""

    pass


class Variable:
    """A base class representing a stateful variable in the framework."""

    def __init__(self, value: object, **kwargs: object) -> None:
        """Initialize."""
        self.value = value


class Param(Variable):
    """A variable representing a trainable parameter."""

    pass


class BatchStat(Variable):
    """A variable representing non-trainable state."""

    pass


class Rng(Variable):
    """A variable holding a random number generator stream."""

    pass


class Dense(Module):
    """A standard linear transformation layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        **kwargs: object,
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias


class Linear(Dense):
    """Alias/variant of standard linear (dense) layer."""

    pass


class LinearGeneral(Module):
    """A general linear transformation layer."""

    def __init__(
        self,
        in_features: Union[int, Sequence[int]],
        out_features: Union[int, Sequence[int]],
        **kwargs: object,
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features


class Einsum(Module):
    """A module that performs a linear transformation using an einsum equation."""

    def __init__(
        self, einsum_str: str, kernel_shape: tuple[int, ...], **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.einsum_str = einsum_str
        self.kernel_shape = kernel_shape


class LoRA(Module):
    """Low-Rank Adaptation injection module."""

    pass


class LoRALinear(Module):
    """A pre-configured Linear layer with LoRA adapters."""

    def __init__(self, in_features: int, out_features: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features


class Conv(Module):
    """A general n-dimensional convolutional layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, ...],
        strides: Optional[tuple[int, ...]] = None,
        padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
        **kwargs: object,
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding


class ConvTranspose(Module):
    """A general n-dimensional transposed convolution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, ...],
        **kwargs: object,
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size


class Embed(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary."""

    def __init__(self, num_embeddings: int, features: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.features = features


class MultiHeadAttention(Module):
    """Standard Multi-Head Attention implementation."""

    def __init__(self, num_heads: int, qkv_features: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qkv_features = qkv_features


class MultiHeadDotProductAttention(Module):
    """Core attention kernel without projection layers."""

    def __init__(self, num_heads: int, qkv_features: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qkv_features = qkv_features


class BatchNorm(Module):
    """Batch Normalization layer."""

    def __init__(
        self, num_features: int, use_running_average: bool = False, **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.num_features = num_features
        self.use_running_average = use_running_average


class LayerNorm(Module):
    """Layer Normalization layer."""

    def __init__(
        self, num_features: int, reduction_axes: int = -1, **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.num_features = num_features
        self.reduction_axes = reduction_axes


class RMSNorm(Module):
    """Root Mean Square Normalization layer."""

    def __init__(self, num_features: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.num_features = num_features


class Dropout(Module):
    """A dropout layer."""

    def __init__(
        self, rate: float, rng_collection: str = "dropout", **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.rate = rate
        self.rng_collection = rng_collection


class Sequential(Module):
    """Applies a sequence of modules sequentially."""

    def __init__(self, *layers: Module, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.layers = layers


class List(Module):
    """A module that holds a list of sub-modules."""

    def __init__(self, modules: Iterable[Module], **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.modules = list(modules)


class Dict(Module):
    """A module that holds a dictionary of sub-modules."""

    def __init__(self, modules: dict[str, Module], **kwargs: object) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.modules = modules


class Jit(Module):
    """JIT-compiles the execution of a sub-module."""

    def __init__(
        self, module_constructor: Callable[..., Module], **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


class Vmap(Module):
    """Vectorizes the execution of a sub-module."""

    def __init__(
        self, module_constructor: Callable[..., Module], **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


class Scan(Module):
    """Loops over a sequence of inputs, maintaining module state."""

    def __init__(
        self, module_constructor: Callable[..., Module], **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


class Remat(Module):
    """Checkpoints a sub-module to save memory during backprop."""

    def __init__(
        self, module_constructor: Callable[..., Module], **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


class Pmap(Module):
    """Parallelizes module execution across multiple devices."""

    def __init__(
        self, module_constructor: Callable[..., Module], **kwargs: object
    ) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
