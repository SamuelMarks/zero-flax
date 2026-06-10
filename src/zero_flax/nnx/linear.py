"""Module docstring."""

from typing import Any, Callable, Sequence, Union, Tuple, Mapping, Optional
from zero_flax.nnx.module import Module
from zero_flax.nnx.state import Param
from zero_jax import numpy as jnp
from zero_flax.nnx import initializers


class Einsum(Module):
    """Docstring."""

    def __init__(
        self,
        einsum_str: str,
        kernel_shape: Tuple[int, ...],
        bias_shape: Optional[Tuple[int, ...]] = None,
        *args,
        **kwargs,
    ):
        """Docstring."""
        super().__init__()
        self.kernel_shape = kernel_shape
        self._is_initializing = False

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        """Docstring."""
        import numpy as np

        return jnp.zeros(np.shape(x)[:-1] + (self.kernel_shape[-1],))


class Linear(Module):
    """Docstring."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: Any = None,
        param_dtype: Any = None,
        precision: Any = None,
        kernel_init: Optional[Callable] = None,
        bias_init: Optional[Callable] = None,
        dot_general: Any = None,
        rngs: Any = None,
    ):
        """Docstring."""
        super().__init__()
        kernel_init = kernel_init or initializers.glorot_uniform()
        bias_init = bias_init or initializers.zeros
        self.kernel = Param(kernel_init(None, (in_features, out_features)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = Param(bias_init(None, (out_features,)))
        self._is_initializing = False

    def __call__(self, inputs: Any) -> Any:
        """Docstring."""
        # Use np.tensordot for arbitrary axes contraction
        import numpy as np

        y = np.tensordot(inputs, self.kernel.value, axes=([-1], [0]))
        if self.use_bias:
            y = np.add(y, self.bias.value)
        return y


class LinearGeneral(Module):
    """Docstring."""

    def __init__(
        self,
        in_features: Union[int, Sequence[int]],
        out_features: Union[int, Sequence[int]],
        *args,
        **kwargs,
    ):
        """Docstring."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._is_initializing = False

    def __call__(self, inputs: Any) -> Any:
        """Docstring."""
        import numpy as np

        _out = (
            [self.out_features]
            if isinstance(self.out_features, int)
            else list(self.out_features)
        )
        return jnp.zeros(np.shape(inputs)[:-1] + tuple(_out))
