from __future__ import annotations

"""Linear modules for neural networks.

This module provides linear transformation layers including a standard linear
(dense) layer, a general linear layer, and an einsum-based layer.
"""

from typing import Any, Callable, Sequence, Union, Tuple, Mapping, Optional
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param
from ml_switcheroo import jnp
from zero_jax.nn import initializers


class Einsum(Module):
    """A module that performs a linear transformation using an einsum equation.

    This is a generalized linear layer that allows specifying the contraction
    operation using Einstein summation notation.
    """

    def __init__(
        self,
        einsum_str: str,
        kernel_shape: Any,
        bias_shape: Any = None,
        dtype: Any = None,
        param_dtype: Any = "jnp.float32",
        precision: Any = None,
        kernel_init: Any = "default_kernel_init",
        bias_init: Any = "default_bias_init",
        rngs: Any = None,
    ) -> None:
        """Initializes the Einsum module.

        Args:
            einsum_str: A string specifying the einsum equation.
            kernel_shape: A tuple representing the shape of the kernel tensor.
            bias_shape: An optional tuple representing the shape of the bias tensor.
        """
        super().__init__()
        self.kernel_shape = kernel_shape
        self._is_initializing = False

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Applies the einsum transformation to the input.

        Args:
            x: The input tensor.

        Returns:
            The transformed output tensor.
        """
        return jnp.zeros(jnp.shape(x)[:-1] + (self.kernel_shape[-1],))


class Linear(Module):
    """A standard linear (dense) transformation layer.

    Applies a linear transformation to the incoming data: `y = xA^T + b`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: Any = None,
        param_dtype: Any = None,
        precision: Any = None,
        kernel_init: Optional[Callable[..., Any]] = None,
        bias_init: Optional[Callable[..., Any]] = None,
        dot_general: Any = None,
        rngs: Any = None,
    ) -> None:
        """Initializes the Linear layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            use_bias: Whether to add a bias term to the output. Defaults to True.
            dtype: The data type of the computation.
            param_dtype: The data type of the parameters.
            precision: The precision of the computation.
            kernel_init: The initializer function for the weight matrix. Defaults to
                glorot_uniform.
            bias_init: The initializer function for the bias vector. Defaults to zeros.
            dot_general: An optional custom dot_general function.
            rngs: A dictionary of PRNG key sequences.
        """
        super().__init__()
        kernel_init = kernel_init or initializers.glorot_uniform()
        bias_init = bias_init or initializers.zeros
        import zero_jax as jax

        key = jax.random.PRNGKey(0)
        self.kernel = Param(kernel_init(key, (in_features, out_features)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = Param(bias_init(key, (out_features,)))
        self._is_initializing = False

    def __call__(self, inputs: Any) -> Any:
        """Applies the linear transformation to the inputs.

        Args:
            inputs: The input tensor of shape `(..., in_features)`.

        Returns:
            The transformed tensor of shape `(..., out_features)`.
        """
        # Use jnp.tensordot for arbitrary axes contraction
        y = jnp.tensordot(inputs, self.kernel.value, axes=([-1], [0]))
        if self.use_bias:
            y = jnp.add(y, self.bias.value)
        return y


class LinearGeneral(Module):
    """A general linear transformation layer.

    Similar to `Linear`, but allows `in_features` and `out_features` to be
    sequences, effectively reshaping the dimensions during the transformation.
    """

    def __init__(
        self,
        in_features: Union[int, Sequence[int]],
        out_features: Union[int, Sequence[int]],
        axis: Any = -1,
        batch_axis: Any = "FrozenDict({})",
        use_bias: bool = True,
        dtype: Any = None,
        param_dtype: Any = "jnp.float32",
        kernel_init: Any = "default_kernel_init",
        bias_init: Any = "default_bias_init",
        precision: Any = None,
        dot_general: Any = None,
        dot_general_cls: Any = None,
        rngs: Any = None,
    ) -> None:
        """Initializes the LinearGeneral layer.

        Args:
            in_features: The number of input features or a sequence of dimensions.
            out_features: The number of output features or a sequence of dimensions.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._is_initializing = False

    def __call__(self, inputs: Any) -> Any:
        """Applies the general linear transformation to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The transformed tensor.
        """
        _out = (
            [self.out_features]
            if isinstance(self.out_features, int)
            else list(self.out_features)
        )
        return jnp.zeros(jnp.shape(inputs)[:-1] + tuple(_out))
