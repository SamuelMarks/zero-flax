"""Missing mock definitions for zero_flax.nnx.

This module provides placeholder definitions and mock implementations for
various components that are normally present in Flax NNX, allowing for
testing and type-checking without needing the actual implementations.
"""

from __future__ import annotations


from zero_flax.nnx.module import Module
from typing import Any, Callable, Sequence, Optional, Iterable, Mapping, Type

DType = Any

Array = Any
ArrayLike = Any


class AxisName:
    """Represents the name of an axis.

    Used to uniquely identify an axis in distributed computation or vectorization.
    """

    pass


class PrecisionLike:
    """Type alias for precision specification.

    Used to specify the precision of numerical operations, such as matrix multiplications.
    """

    pass


class PaddingLike:
    """Type alias for padding specification.

    Used to specify how padding should be applied in operations like convolutions.
    """

    pass


class Dtype:
    """Represents a data type.

    Used to specify the type of numerical data, such as float32, int32, etc.
    """

    pass


class Shape:
    """Represents the shape of an array.

    Specifies the dimensions of an n-dimensional array.
    """

    pass


class Axes:
    """Represents multiple axes.

    Used to specify a collection of axes for operations like reduction or transposition.
    """

    pass


class Size:
    """Represents the size of a dimension.

    Used to specify the number of elements along a particular axis.
    """

    pass


class Axis:
    """Represents a single axis.

    Used to identify a specific dimension in an array.
    """

    pass


class DotGeneralT:
    """Type alias for dot_general function signature.

    Specifies the signature for a generalized dot product operation.
    """

    pass


class MaxFun:
    """Type alias for a maximum function.

    Used to specify a function that computes the maximum value.
    """

    pass


class filterlib:
    """Mock for filterlib module containing filter utilities."""

    class Filter:
        """Represents a filter criterion.

        Used to filter collections based on specific properties or conditions.
        """

        pass


class rnglib:
    """Mock for rnglib module containing random number generator utilities."""

    class Rngs:
        """Manages random number generator state.

        Provides a structured way to pass and update PRNG keys across modules.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initializes the Rngs instance.

            Args:
                *args: Positional arguments for initialization.
                **kwargs: Keyword arguments for initialization.
            """
            pass

        pass


class variables:
    """Mock for variables module containing state variables."""

    class Variable:
        """Represents a state variable within a module.

        Used to store mutable state such as parameters or running statistics.
        """

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
    """Type alias for a variable initializer function.

    Specifies the signature for functions that initialize parameters.
    """

    pass


class initializers:
    """Mock for initializers module containing initialization functions."""

    zeros = None
    ones = None
    zeros_init = None
    ones_init = None


class BatchNorm(Module):
    """Batch normalization module.

    Applies batch normalization over a mini-batch of inputs.
    """

    def __init__(
        self,
        num_features: int,
        use_running_average: bool = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-05,
        dtype: Optional[Dtype] = None,
        param_dtype: str | type[Any] | DType | Any | Any = None,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Any | Callable[..., Any] = None,
        scale_init: Any | Callable[..., Any] = None,
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: Any = None,
    ) -> None:
        """Initializes the BatchNorm module.

        Args:
            num_features: Number of features in the input.
            use_running_average: Whether to use running statistics.
            axis: The axis that should be normalized.
            momentum: Momentum for the running average.
            epsilon: A small float added to variance to avoid dividing by zero.
            dtype: The dtype of the computation.
            param_dtype: The dtype passed to parameter initializers.
            use_bias: If True, adds a learnable bias to the output.
            use_scale: If True, multiplies by a learnable scale.
            bias_init: Initializer for bias.
            scale_init: Initializer for scale.
            axis_name: The axis name used to combine batch statistics.
            axis_index_groups: Groups of axis indices.
            use_fast_variance: If True, uses a faster variance computation.
            rngs: Rngs instance for initializing parameters.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Applies batch normalization to the input.

        Args:
            *args: Input array(s).
            **kwargs: Additional keyword arguments.

        Returns:
            The normalized output array.
        """
        pass


class ConvTranspose(Module):
    """Transposed convolution module.

    Applies a transposed convolution operation over an input signal.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] | None = None,
        padding: Any = "SAME",
        kernel_dilation: int | Sequence[int] | None = None,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: Dtype | None = None,
        param_dtype: Any = None,
        precision: PrecisionLike | None = None,
        kernel_init: Any = None,
        bias_init: Any = None,
        transpose_kernel: bool = False,
        rngs: Any = None,
    ) -> None:
        """Initializes the ConvTranspose module.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            kernel_size: Spatial dimensions of the kernel.
            strides: A sequence of `n` integers representing the spatial strides.
            padding: Padding specification.
            kernel_dilation: Sequence of `n` integers specifying kernel dilation.
            use_bias: Whether to add a bias term to the output.
            mask: Optional mask array.
            dtype: The dtype of the computation.
            param_dtype: The dtype passed to parameter initializers.
            precision: Numerical precision of the computation.
            kernel_init: Initializer for the convolutional kernel.
            bias_init: Initializer for the bias.
            transpose_kernel: If True, transposes the kernel dimensions.
            rngs: Rngs instance for initializing parameters.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Applies a transposed convolution to the inputs.

        Args:
            *args: Input array(s).
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the transposed convolution.
        """
        pass


class Dropout(Module):
    """Dropout module.

    Applies dropout to inputs, randomly setting a fraction of them to zero.
    """

    def __init__(
        self,
        rate: float,
        broadcast_dims: Sequence[int] = (),
        deterministic: bool = False,
        rng_collection: str = "dropout",
        rngs: Any | None = None,
    ):
        """Initializes the Dropout module.

        Args:
            rate: The probability of an element to be zeroed.
            broadcast_dims: Dimensions that will share the same dropout mask.
            deterministic: If True, dropout is turned off.
            rng_collection: The rng collection name to use for generating a mask.
            rngs: Rngs instance used to generate random keys.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Applies dropout to the inputs.

        Args:
            *args: Input array(s).
            **kwargs: Additional keyword arguments.

        Returns:
            The input array with dropout applied.
        """
        pass


class Jit(Module):
    """Jit module.

    Wraps a module constructor and applies JIT compilation.
    """

    def __init__(
        self,
        module_constructor: Callable[..., M],
        in_shardings: Any = None,
        out_shardings: Any = None,
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
        module_init_args: Optional[tuple[Any, ...]] = None,
        module_init_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initializes the Jit module.

        Args:
            module_constructor: Function or class to instantiate the underlying module.
            in_shardings: Sharding specifications for the inputs.
            out_shardings: Sharding specifications for the outputs.
            static_argnums: Indices of static positional arguments.
            static_argnames: Names of static keyword arguments.
            donate_argnums: Indices of positional arguments to donate.
            donate_argnames: Names of keyword arguments to donate.
            keep_unused: If False, prunes computations that are not used.
            device: Device to compile the computation for.
            backend: Backend to compile the computation for.
            inline: If True, inlines the computation into the caller.
            abstracted_axes: Abstraction axes for the computation.
            donate_state: If True, the internal state is donated.
            constrain_state: Constraints on the internal state.
            module_init_args: Positional arguments for module_constructor.
            module_init_kwargs: Keyword arguments for module_constructor.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the JIT-compiled underlying module.

        Args:
            *args: Positional arguments passed to the module.
            **kwargs: Keyword arguments passed to the module.

        Returns:
            The result of the JIT-compiled computation.
        """
        pass


class LoRA(Module):
    """Low-Rank Adaptation (LoRA) module.

    A standalone LoRA layer that computes a low-rank update.
    """

    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        base_module: Optional[Module] = None,
        dtype: Optional[Dtype] = None,
        param_dtype: str | type[Any] | DType | Any | Any = None,
        kernel_init: Any | Callable[..., Any] = None,
        lora_param_type: Optional[Type[variables.Variable]] = None,
        rngs: Any = None,
    ) -> None:
        """Initializes the LoRA module.

        Args:
            in_features: Number of input features.
            lora_rank: The rank of the low-rank adaptation.
            out_features: Number of output features.
            base_module: The base module to which LoRA is applied.
            dtype: The dtype of the computation.
            param_dtype: The dtype passed to parameter initializers.
            kernel_init: Initializer for the LoRA matrices.
            lora_param_type: Variable type to use for LoRA parameters.
            rngs: Rngs instance for initializing parameters.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Applies the low-rank adaptation to the input.

        Args:
            *args: Input array(s).
            **kwargs: Additional keyword arguments.

        Returns:
            The output array after applying the LoRA computation.
        """
        pass


class LoRALinear(Module):
    """Linear module with Low-Rank Adaptation (LoRA).

    An `nnx.Linear` layer that integrates a LoRA update.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_dtype: Optional[Dtype] = None,
        lora_param_dtype: str | type[Any] | DType | Any | Any = None,
        lora_kernel_init: Any | Callable[..., Any] = None,
        lora_param_type: Optional[Type[variables.Variable]] = None,
        rngs: Any = None,
        kwargs: Any = {},
        use_bias: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Optional[Dtype] = None,
        precision: Optional[PrecisionLike] = None,
        kernel_init: Optional[Initializer] = None,
        bias_init: Optional[Initializer] = None,
        dot_general: Optional[DotGeneralT] = None,
    ) -> None:
        """Initializes the LoRALinear module.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            lora_rank: The rank of the low-rank adaptation.
            lora_dtype: The dtype of the LoRA computation.
            lora_param_dtype: The dtype passed to LoRA parameter initializers.
            lora_kernel_init: Initializer for the LoRA matrices.
            lora_param_type: Variable type to use for LoRA parameters.
            rngs: Rngs instance for initializing parameters.
            kwargs: Additional keyword arguments.
            use_bias: Whether to add a bias term to the linear transformation.
            dtype: The dtype of the computation.
            param_dtype: The dtype passed to parameter initializers.
            precision: Numerical precision of the computation.
            kernel_init: Initializer for the linear kernel.
            bias_init: Initializer for the bias.
            dot_general: The dot product function to use.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Applies the linear transformation with LoRA update.

        Args:
            *args: Input array(s).
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the linear transformation plus the LoRA update.
        """
        pass


class MultiHeadAttention(Module):
    """Multi-head attention module.

    Applies multi-head dot-product attention over an input sequence.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        dtype: Dtype | None = None,
        param_dtype: Any = None,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: Optional[PrecisionLike] = None,
        kernel_init: Any = None,
        out_kernel_init: Initializer | None = None,
        bias_init: Any = None,
        out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        attention_fn: Any = "(dot_product_attention)",
        decode: bool | None = None,
        normalize_qk: bool = False,
        qkv_dot_general: DotGeneralT | None = None,
        out_dot_general: DotGeneralT | None = None,
        qkv_dot_general_cls: Any = None,
        out_dot_general_cls: Any = None,
        rngs: Any = None,
    ):
        """Initializes the MultiHeadAttention module.

        Args:
            num_heads: Number of attention heads.
            in_features: Number of input features.
            qkv_features: Number of features for queries, keys, and values.
            out_features: Number of output features.
            dtype: The dtype of the computation.
            param_dtype: The dtype passed to parameter initializers.
            broadcast_dropout: Whether to broadcast dropout across spatial dimensions.
            dropout_rate: Dropout rate on attention weights.
            deterministic: If True, disables dropout.
            precision: Numerical precision of the computation.
            kernel_init: Initializer for QKV kernels.
            out_kernel_init: Initializer for the output projection kernel.
            bias_init: Initializer for QKV biases.
            out_bias_init: Initializer for the output projection bias.
            use_bias: Whether to use biases in projections.
            attention_fn: The attention function to use.
            decode: Whether to prepare and use an autoregressive cache.
            normalize_qk: Whether to apply layer normalization to Q and K.
            qkv_dot_general: Custom dot general function for QKV projections.
            out_dot_general: Custom dot general function for output projection.
            qkv_dot_general_cls: Class for QKV dot general function.
            out_dot_general_cls: Class for output dot general function.
            rngs: Rngs instance for initializing parameters.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Applies multi-head attention to the inputs.

        Args:
            *args: Positional arguments including queries, keys, and values.
            **kwargs: Keyword arguments for the attention computation.

        Returns:
            The output of the attention mechanism.
        """
        pass


class Pmap(Module):
    """Pmap module.

    Wraps a module constructor and applies parallel mapping (pmap) over devices.
    """

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
        state_axes: Optional[Mapping[filterlib.Filter, int]] = None,
        split_rngs: Any = Ellipsis,
        transform_metadata: Optional[Mapping[str, Any]] = None,
        module_init_args: Optional[tuple[Any, ...]] = None,
        module_init_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initializes the Pmap module.

        Args:
            module_constructor: Function or class to instantiate the underlying module.
            axis_name: The name of the mapped axis.
            in_axes: Axis specification for mapping the inputs.
            out_axes: Axis specification for mapping the outputs.
            static_broadcasted_argnums: Indices of arguments that are static.
            devices: Devices over which to parallelize.
            backend: The backend to use.
            axis_size: The size of the mapped axis.
            donate_argnums: Indices of arguments to donate.
            global_arg_shapes: Global shapes of arguments.
            in_axes_kwargs: Axis specification for mapping keyword arguments.
            state_axes: Axes specification for the module state.
            split_rngs: Rng splitting specification.
            transform_metadata: Metadata for the transformation.
            module_init_args: Positional arguments for module_constructor.
            module_init_kwargs: Keyword arguments for module_constructor.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the mapped module across devices.

        Args:
            *args: Positional arguments to the mapped module.
            **kwargs: Keyword arguments to the mapped module.

        Returns:
            The result of the mapped computation.
        """
        pass


class Remat(Module):
    """Rematerialization (checkpointing) module.

    Recomputes activations during the backward pass to save memory.
    """

    def __init__(
        self,
        module_constructor: Callable[..., M],
        prevent_cse: bool = True,
        static_argnums: int | tuple[int, ...] = (),
        policy: Callable[..., bool] | None = None,
        module_init_args: Optional[tuple[Any, ...]] = None,
        module_init_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initializes the Remat module.

        Args:
            module_constructor: Function or class to instantiate the underlying module.
            prevent_cse: Whether to prevent common subexpression elimination across boundaries.
            static_argnums: Indices of arguments that are static.
            policy: Policy function specifying which operations to checkpoint.
            module_init_args: Positional arguments for module_constructor.
            module_init_kwargs: Keyword arguments for module_constructor.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the module with rematerialization applied.

        Args:
            *args: Positional arguments passed to the underlying module.
            **kwargs: Keyword arguments passed to the underlying module.

        Returns:
            The output of the module computation.
        """
        pass


class Scan(Module):
    """Scan module.

    Applies a function repeatedly while carrying along state.
    """

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
        state_axes: Optional[Mapping[filterlib.Filter, int]] = None,
        split_rngs: Any = Ellipsis,
        transform_metadata: Optional[Mapping[str, Any]] = None,
        scan_output: bool = True,
        module_init_args: Optional[tuple[Any, ...]] = None,
        module_init_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initializes the Scan module.

        Args:
            module_constructor: Function or class to instantiate the underlying module.
            length: The number of loop iterations.
            reverse: If True, iterates over the sequences in reverse.
            unroll: Number of loop iterations to unroll.
            _split_transpose: Internal detail for transpose behavior.
            in_axes: Specifies axes to map over for positional arguments.
            in_axes_kwargs: Specifies axes to map over for keyword arguments.
            out_axes: Specifies axes to map over for the output.
            carry_argnum: The index of the argument representing the loop carry.
            state_axes: Axes specification for the module state.
            split_rngs: Rng splitting specification.
            transform_metadata: Metadata for the transformation.
            scan_output: Whether to accumulate outputs along the scan axis.
            module_init_args: Positional arguments for module_constructor.
            module_init_kwargs: Keyword arguments for module_constructor.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the scanned module over the sequence.

        Args:
            *args: Positional arguments including the loop carry and mapped inputs.
            **kwargs: Keyword arguments passed to the module.

        Returns:
            A tuple of the final loop carry and stacked outputs.
        """
        pass


class Vmap(Module):
    """Vmap module.

    Wraps a module constructor and applies vectorization mapping (vmap).
    """

    def __init__(
        self,
        module_constructor: Callable[..., M],
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
        axis_name: AxisName | None = None,
        axis_size: int | None = None,
        spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
        in_axes_kwargs: Any = 0,
        state_axes: Optional[Mapping[filterlib.Filter, int]] = None,
        split_rngs: Any = Ellipsis,
        transform_metadata: Optional[Mapping[str, Any]] = None,
        module_init_args: Optional[tuple[Any, ...]] = None,
        module_init_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initializes the Vmap module.

        Args:
            module_constructor: Function or class to instantiate the underlying module.
            in_axes: Axis specification for vectorizing the inputs.
            out_axes: Axis specification for vectorizing the outputs.
            axis_name: The name of the vectorized axis.
            axis_size: The size of the vectorized axis.
            spmd_axis_name: SPMD axis names for distributed execution.
            in_axes_kwargs: Axis specification for vectorizing keyword arguments.
            state_axes: Axes specification for the module state.
            split_rngs: Rng splitting specification.
            transform_metadata: Metadata for the transformation.
            module_init_args: Positional arguments for module_constructor.
            module_init_kwargs: Keyword arguments for module_constructor.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the vectorized module.

        Args:
            *args: Positional arguments to the vectorized module.
            **kwargs: Keyword arguments to the vectorized module.

        Returns:
            The result of the vectorized computation.
        """
        pass
