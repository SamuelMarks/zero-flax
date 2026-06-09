from typing import Any, Callable, Sequence, Union, Tuple, Mapping
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param
from zero_jax import numpy as jnp
import numpy as np


class Einsum(Module):
    """An einsum transformation with learnable kernel and bias."""

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
        self.einsum_str = einsum_str
        self.kernel_shape = kernel_shape
        self.bias_shape = bias_shape
        k1 = np.array([0, 0]) if rngs is None else rngs
        self.kernel = Param(kernel_init(k1, kernel_shape))
        if bias_shape is not None:
            self.bias = Param(bias_init(k1, bias_shape))
        else:
            self.bias = None
        super().__setattr__("_is_initializing", False)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        # Mock with standard einsum
        y = jnp.einsum(self.einsum_str, x, self.kernel.value)
        if self.bias is not None:
            y = jnp.add(y, self.bias.value)
        return y


class Linear(Module):
    """A linear transformation applied over the last dimension of the input."""

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
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        k1 = np.array([0, 0]) if rngs is None else rngs
        self.kernel = Param(kernel_init(k1, (in_features, out_features)))
        if use_bias:
            self.bias = Param(bias_init(k1, (out_features,)))
        else:
            self.bias = None
        super().__setattr__("_is_initializing", False)

    def __call__(self, inputs: Any) -> Any:
        y = np.dot(inputs, self.kernel.value)
        if self.use_bias:
            y = jnp.add(y, self.bias.value)
        return y


class LinearGeneral(Module):
    """A linear transformation with flexible axes."""

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
        self.use_bias = use_bias
        in_feat = (in_features,) if isinstance(in_features, int) else tuple(in_features)
        out_feat = (
            (out_features,) if isinstance(out_features, int) else tuple(out_features)
        )
        k1 = np.array([0, 0]) if rngs is None else rngs
        self.kernel = Param(kernel_init(k1, in_feat + out_feat))
        if use_bias:
            self.bias = Param(bias_init(k1, out_feat))
        else:
            self.bias = None
        super().__setattr__("_is_initializing", False)

    def __call__(self, inputs: Any) -> Any:
        # Mock behavior
        y = jnp.add(np.dot(inputs, self.kernel.value), 0.0)
        if self.use_bias:
            y = jnp.add(y, self.bias.value)
        return y
