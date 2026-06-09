from zero_flax.nnx.module import Module
from typing import Iterable, Callable, Any, TypeVar

A = TypeVar("A")


class Dict(Module):
    """Base class for all neural network modules."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for i, arg in enumerate(args):
            setattr(self, str(i), arg)
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__setattr__("_is_initializing", False)

    def __call__(self, *args, **kwargs):
        pass


class List(Module):
    """Base class for all neural network modules."""

    def __init__(self, elems: Iterable[A] = (), /):
        super().__init__()
        for i, elem in enumerate(elems):
            setattr(self, str(i), elem)
        self.length = len(list(elems))
        super().__setattr__("_is_initializing", False)

    def __iter__(self):
        for i in range(self.length):
            yield getattr(self, str(i))

    def __getitem__(self, idx):
        return getattr(self, str(idx))


class Sequential(Module):
    """Base class for all neural network modules."""

    def __init__(self, *fns: Callable[..., Any]):
        super().__init__()
        for i, fn in enumerate(fns):
            setattr(self, str(i), fn)
        self.length = len(fns)
        super().__setattr__("_is_initializing", False)

    def __call__(self, x, *args, **kwargs):
        for i in range(self.length):
            fn = getattr(self, str(i))
            if isinstance(fn, type(self)) or hasattr(fn, "__call__"):
                x = fn(x, *args, **kwargs)
            else:
                x = fn(x)
        return x
