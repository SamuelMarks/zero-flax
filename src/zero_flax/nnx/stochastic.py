"""Module docstring."""

from zero_flax.nnx.module import Module


class Dropout(Module):
    """Docstring."""

    def __init__(self, rate: float, *args, **kwargs):
        """Docstring."""
        super().__init__()
        self.rate = rate
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        """Docstring."""
        return x
