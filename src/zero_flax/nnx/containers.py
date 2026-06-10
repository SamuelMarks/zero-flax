"""Module docstring."""

from zero_flax.nnx.module import Module
from typing import Iterable, Callable, Any, TypeVar

A = TypeVar("A")


class Dict(Module, dict):
    """Base class for all neural network modules."""

    def __init__(self, *args, **kwargs):
        """Docstring."""
        super().__init__()
        dict.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Docstring."""
        return None


class List(Module, list):
    """Base class for all neural network modules."""

    def __init__(self, elems: Iterable[A] = (), /):
        """Docstring."""
        super().__init__()
        list.__init__(self, elems)

    def __iter__(self):
        """Docstring."""
        return list.__iter__(self)

    def __getitem__(self, idx):
        """Docstring."""
        return list.__getitem__(self, idx)


class Sequential(Module):
    """Base class for all neural network modules."""

    def __init__(self, *fns: Callable[..., Any]):
        """Docstring."""
        super().__init__()
        self.fns = fns

    def __call__(self, x, *args, **kwargs):
        """Docstring."""
        for fn in self.fns:
            x = fn(x, *args, **kwargs)
        return x
