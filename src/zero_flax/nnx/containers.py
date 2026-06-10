from zero_flax.nnx.module import Module
from typing import Iterable, Callable, Any, TypeVar

A = TypeVar("A")


class Dict(Module):
    """Base class for all neural network modules."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class List(Module):
    """Base class for all neural network modules."""

    def __init__(self, elems: Iterable[A] = (), /):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, idx):
        pass


class Sequential(Module):
    """Base class for all neural network modules."""

    def __init__(self, *fns: Callable[..., Any]):
        pass

    def __call__(self, x, *args, **kwargs):
        pass
