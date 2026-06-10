from typing import Any, Callable, Sequence, Union, Tuple, Mapping
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param
from zero_jax import numpy as jnp
from zero_flax.nnx import initializers


class Einsum(Module):
    def __init__(
        self,
        einsum_str: str,
        kernel_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...] = None,
        dtype: Any = None,
        param_dtype: Any = None,
        precision: Any = None,
        kernel_init: Callable = None,
        bias_init: Callable = None,
        rngs: Any = None,
    ):
        super().__init__()
        self.kernel_shape = kernel_shape
        self._is_initializing = False

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return jnp.dot(x, initializers.ones(None, self.kernel_shape))


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: Any = None,
        param_dtype: Any = None,
        precision: Any = None,
        kernel_init: Callable = None,
        bias_init: Callable = None,
        dot_general: Any = None,
        rngs: Any = None,
    ):
        super().__init__()
        kernel_init = kernel_init or initializers.glorot_uniform()
        bias_init = bias_init or initializers.zeros
        self.kernel = Param(kernel_init(None, (in_features, out_features)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = Param(bias_init(None, (out_features,)))
        self._is_initializing = False

    def __call__(self, inputs: Any) -> Any:
        y = jnp.dot(inputs, self.kernel.value)
        if self.use_bias:
            y = jnp.add(y, self.bias.value)
        return y


class LinearGeneral(Module):
    def __init__(
        self,
        in_features: Union[int, Sequence[int]],
        out_features: Union[int, Sequence[int]],
        axis: Union[int, Sequence[int]] = -1,
        batch_axis: Mapping[int, int] = (),
        use_bias: bool = True,
        dtype: Any = None,
        param_dtype: Any = None,
        kernel_init: Callable = None,
        bias_init: Callable = None,
        precision: Any = None,
        dot_general: Any = None,
        dot_general_cls: Any = None,
        rngs: Any = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._is_initializing = False

    def __call__(self, inputs: Any) -> Any:
        return jnp.dot(
            inputs, initializers.ones(None, (self.in_features, self.out_features))
        )
