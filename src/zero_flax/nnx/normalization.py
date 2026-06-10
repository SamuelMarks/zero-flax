"""Module docstring."""

from zero_flax.nnx.module import Module


class BatchNorm(Module):
    """Docstring."""

    def __init__(self, features: int, *args, **kwargs):
        """Docstring."""
        super().__init__()
        self.features = features
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        """Docstring."""
        return x


class LayerNorm(Module):
    """Docstring."""

    def __init__(self, features: int, *args, **kwargs):
        """Docstring."""
        super().__init__()
        self.features = features
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        """Docstring."""
        return x


class RMSNorm(Module):
    """Docstring."""

    def __init__(self, features: int, *args, **kwargs):
        """Docstring."""
        super().__init__()
        self.features = features
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        """Docstring."""
        return x
