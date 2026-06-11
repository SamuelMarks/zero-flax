"""Container modules for neural networks.

This module provides standard container modules like Dict, List, and Sequential
that inherit from Module and can be used to group other modules or elements.
"""

from zero_flax.nnx.module import Module
from typing import Iterable, Callable, Any, TypeVar

A = TypeVar("A")


class Dict(Module, dict[str, Any]):
    """A dictionary-like container module.

    This module acts as a standard Python dictionary but also registers its
    contents as submodules if they are Module instances.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the Dict container.

        Args:
            *args: Positional arguments passed to the underlying dict constructor.
            **kwargs: Keyword arguments passed to the underlying dict constructor.
        """
        super().__init__()
        dict.__init__(self, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allows calling the Dict module directly.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Always returns None.
        """
        return None


class List(Module, list[Any]):
    """A list-like container module.

    This module acts as a standard Python list but also registers its
    contents as submodules if they are Module instances.
    """

    def __init__(self, elems: Iterable[A] = (), /):
        """Initializes the List container.

        Args:
            elems: An iterable of elements to initialize the list with.
        """
        super().__init__()
        list.__init__(self, elems)

    def __iter__(self) -> Any:
        """Returns an iterator over the elements in the List.

        Returns:
            An iterator object.
        """
        return list.__iter__(self)

    def __getitem__(self, idx: Any) -> Any:
        """Retrieves an element at the given index.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The element at the specified index.
        """
        return list.__getitem__(self, idx)


class Sequential(Module):
    """A sequential container module.

    This module sequences together a list of callable objects (usually Modules),
    passing the output of one as the input to the next.
    """

    def __init__(self, *fns: Callable[..., Any]) -> None:
        """Initializes the Sequential container.

        Args:
            *fns: A variable number of callable objects to sequence together.
        """
        super().__init__()
        self.fns = fns

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Applies the sequenced functions to the input.

        Args:
            x: The initial input tensor or value.
            *args: Additional positional arguments passed to each function.
            **kwargs: Additional keyword arguments passed to each function.

        Returns:
            The final output after applying all functions in sequence.
        """
        for fn in self.fns:
            x = fn(x, *args, **kwargs)
        return x
