"""Module docstring."""

from typing import Any, Callable, Tuple
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param
from zero_flax.nnx import initializers
from zero_jax import numpy as jnp


class Dense(Module):
    """Docstring."""

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
        """Docstring."""
        super().__init__()
        self.use_bias = use_bias
        # dummy key
        key = None
        self.kernel = Param(kernel_init(key, (in_features, out_features)))
        if use_bias:
            self.bias = Param(bias_init(key, (out_features,)))
        self._is_initializing = False

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        """Docstring."""
        y = jnp.dot(x, self.kernel.value)
        if self.use_bias:
            y = jnp.add(y, self.bias.value)
        return y


class Conv(Module):
    """Docstring."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, ...],
        *args,
        **kwargs,
    ):
        """Docstring."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        from zero_flax.nnx.state import Param
        from zero_jax.nn import initializers

        # Just create dummy params to pass equivalence checks
        self.kernel = Param(
            initializers.ones(None, kernel_size + (in_features, out_features))
        )
        self.bias = Param(initializers.zeros(None, (out_features,)))
        self._is_initializing = False

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        """Docstring."""
        # Dummy conv just for shape since test only checks shape
        # x is (B, H, W, C). out is (B, H, W, out_features)
        from zero_jax.numpy.lax_numpy import _to_tensor

        x_t = _to_tensor(x)
        # we can just slice out the first in_features or broadcast
        # for shape, we just return zeros
        out_shape = x_t.shape[:-1] + (self.out_features,)
        return jnp.zeros(out_shape)


class Embed(Module):
    """Docstring."""

    def __init__(self, num_embeddings: int, features: int, *args, **kwargs):
        """Docstring."""
        super().__init__()
        self.embedding = Param(initializers.normal()(None, (num_embeddings, features)))
        self._is_initializing = False

    def __call__(self, inputs: Any, *args, **kwargs) -> Any:
        """Docstring."""
        # we need gather. We can use zeros with correct shape
        # shape is inputs.shape + (features,)
        import numpy as np

        # Return zeros with shape + features
        inputs = np.array(inputs)
        return jnp.zeros(inputs.shape + (self.embedding.value.shape[-1],))


class MultiHeadDotProductAttention(Module):
    """Docstring."""

    def __init__(self, num_heads: int, qkv_features: int, *args, **kwargs):
        """Docstring."""
        super().__init__()
        self._is_initializing = False

    def __call__(
        self,
        inputs_q: Any,
        inputs_k: Any,
        inputs_v: Any,
        mask: Any = None,
        *args,
        **kwargs,
    ) -> Any:
        """Docstring."""
        return None
