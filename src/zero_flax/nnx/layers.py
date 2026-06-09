from typing import Any, Callable, Tuple
import numpy as np
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param
from zero_flax.nnx import initializers


class Dense(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        kernel_init: Callable = initializers.glorot_uniform(),
        bias_init: Callable = initializers.zeros,
        rngs: Any = None,
        *args,
        **kwargs,
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

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        y = np.einsum("...i,ij->...j", x, self.kernel.value)
        if self.use_bias:
            y = np.add(y, self.bias.value)
        return y


class Conv(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, ...],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.kernel = Param(np.ones(kernel_size + (in_features, out_features)))
        self.bias = Param(np.zeros((out_features,)))
        super().__setattr__("_is_initializing", False)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        out_shape = x.shape[:-1] + (self.out_features,)
        return np.add(np.zeros(out_shape), self.bias.value)


class Embed(Module):
    def __init__(self, num_embeddings: int, features: int, *args, **kwargs):
        super().__init__()
        self.embedding = Param(np.random.normal(0, 0.1, (num_embeddings, features)))
        super().__setattr__("_is_initializing", False)

    def __call__(self, inputs: Any, *args, **kwargs) -> Any:
        return self.embedding.value[inputs]


class MultiHeadDotProductAttention(Module):
    def __init__(self, num_heads: int, qkv_features: int, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_features = qkv_features
        self.dense = Dense(qkv_features, qkv_features)
        super().__setattr__("_is_initializing", False)

    def __call__(
        self,
        inputs_q: Any,
        inputs_k: Any,
        inputs_v: Any,
        mask: Any = None,
        *args,
        **kwargs,
    ) -> Any:
        return self.dense(inputs_q)
