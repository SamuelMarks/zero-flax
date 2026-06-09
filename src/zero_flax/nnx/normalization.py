from typing import Any, Callable
import numpy as np
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param, BatchStat
from zero_flax.nnx import initializers


class BatchNorm(Module):
    """BatchNorm Module."""

    def __init__(
        self,
        num_features: int,
        use_running_average: bool = False,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        dtype: Any = None,
        param_dtype: Any = None,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Callable = initializers.zeros,
        scale_init: Callable = initializers.ones,
        axis_name: Any = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: Any = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.use_running_average = use_running_average
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale

        k1 = np.array([0, 0]) if rngs is None else rngs

        if use_scale:
            self.scale = Param(scale_init(k1, (num_features,)))
        else:
            self.scale = None

        if use_bias:
            self.bias = Param(bias_init(k1, (num_features,)))
        else:
            self.bias = None

        self.mean = BatchStat(np.zeros((num_features,)))
        self.var = BatchStat(np.ones((num_features,)))
        super().__setattr__("_is_initializing", False)

    def __call__(self, x: Any, use_running_average: bool = None) -> Any:
        is_training = not (
            self.use_running_average
            if use_running_average is None
            else use_running_average
        )

        if is_training:
            # compute mean and var over all axes except self.axis
            reduce_axes = tuple(
                i for i in range(x.ndim) if i != self.axis and i != x.ndim + self.axis
            )
            mean = np.mean(x, axis=reduce_axes, keepdims=True)
            var = np.var(x, axis=reduce_axes, keepdims=True)

            # update running stats
            self.mean.value = (
                self.momentum * self.mean.value + (1 - self.momentum) * mean.squeeze()
            )
            self.var.value = (
                self.momentum * self.var.value + (1 - self.momentum) * var.squeeze()
            )
        else:
            mean = self.mean.value
            var = self.var.value
            # broadcast to match x shape
            shape = [1] * x.ndim
            shape[self.axis] = self.num_features
            mean = mean.reshape(shape)
            var = var.reshape(shape)

        y = (x - mean) / np.sqrt(var + self.epsilon)

        if self.use_scale:
            shape = [1] * x.ndim
            shape[self.axis] = self.num_features
            y = y * self.scale.value.reshape(shape)

        if self.use_bias:
            shape = [1] * x.ndim
            shape[self.axis] = self.num_features
            y = y + self.bias.value.reshape(shape)

        return y


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-6,
        dtype: Any = None,
        param_dtype: Any = None,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Callable = initializers.zeros,
        scale_init: Callable = initializers.ones,
        reduction_axes: Any = -1,
        feature_axes: Any = -1,
        axis_name: Any = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: Any = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale

        k1 = np.array([0, 0]) if rngs is None else rngs

        if use_scale:
            self.scale = Param(scale_init(k1, (num_features,)))
        else:
            self.scale = None

        if use_bias:
            self.bias = Param(bias_init(k1, (num_features,)))
        else:
            self.bias = None

        super().__setattr__("_is_initializing", False)

    def __call__(self, x: Any) -> Any:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + self.epsilon)

        if self.use_scale:
            y = y * self.scale.value

        if self.use_bias:
            y = y + self.bias.value

        return y


class RMSNorm(Module):
    """RMS Layer normalization."""

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-6,
        dtype: Any = None,
        param_dtype: Any = None,
        use_scale: bool = True,
        scale_init: Callable = initializers.ones,
        reduction_axes: Any = -1,
        feature_axes: Any = -1,
        axis_name: Any = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: Any = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.use_scale = use_scale

        k1 = np.array([0, 0]) if rngs is None else rngs

        if use_scale:
            self.scale = Param(scale_init(k1, (num_features,)))
        else:
            self.scale = None

        super().__setattr__("_is_initializing", False)

    def __call__(self, x: Any) -> Any:
        var = np.mean(np.square(x), axis=-1, keepdims=True)
        y = x / np.sqrt(var + self.epsilon)

        if self.use_scale:
            y = y * self.scale.value

        return y
