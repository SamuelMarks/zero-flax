"""Container modules (`Dict`, `List`, `Sequential`) for the zero-flax NNX API.

This module implements standard container modules that inherit from `Module`
and automatically register child `Module` and `Variable` entries into the
parent module registries without dynamic reflection or `Any`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Generic, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

from typing_extensions import TypeAlias, TypeGuard

from zero_flax.nnx.module import Module
from zero_flax.nnx.variables import Variable, VariableValue
from zero_flax.types import ArrayLike, JaxArray


@runtime_checkable
class LayerCallableProtocol(Protocol):
    """Protocol for sequential layer callables accepting an input array."""

    def __call__(
        self,
        x: ArrayLike,
        *args: ArrayLike,
        **kwargs: ArrayLike,
    ) -> ArrayLike:
        """Transform ``x`` and return an ``ArrayLike`` output.

        Args:
            x: Input array or scalar.
            *args: Optional additional positional arguments.
            **kwargs: Optional additional keyword arguments.

        Returns:
            Transformed output.
        """


ContainerItem: TypeAlias = Union[
    Module,
    Variable[VariableValue],
    LayerCallableProtocol,
    JaxArray,
    ArrayLike,
    int,
    float,
    str,
    bool,
    None,
]
ItemT = TypeVar("ItemT", bound=ContainerItem)


class Dict(Module, Generic[ItemT]):
    """Dictionary container module registering ``Module`` and ``Variable`` values in NNX state.

    Attributes:
        _data: Internal dictionary mapping string keys to container items.
    """

    _data: dict[str, ItemT]

    def __init__(
        self,
        *args: Union[Mapping[str, ItemT], Iterable[Tuple[str, ItemT]]],
        **kwargs: ItemT,
    ) -> None:
        """Initialize a Dict module from an optional mapping/iterable and keyword entries.

        Args:
            *args: At most one mapping or iterable of ``(key, value)`` pairs.
            **kwargs: Additional key-value entries to store in the container.
        """
        super().__init__()
        self._data = {}
        for arg in args:
            if isinstance(arg, Mapping):
                for k, v in arg.items():
                    self[k] = v
            else:
                for k, v in arg:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v
        self.finish_init()

    def __setitem__(self, key: str, value: ItemT) -> None:
        """Store ``value`` under ``key`` and register it if it is a ``Module`` or ``Variable``.

        Args:
            key: String dictionary key.
            value: Item to store.
        """
        self._ensure_registries()
        self._data[key] = value
        if isinstance(value, Module):
            self.register_child(str(key), value)
        elif isinstance(value, Variable):
            self.register_variable(str(key), value)

    def __getitem__(self, key: str) -> ItemT:
        """Retrieve the item stored under ``key``.

        Args:
            key: String dictionary key.

        Returns:
            Stored item.
        """
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check whether ``key`` is present in the Dict container.

        Args:
            key: String dictionary key to check.

        Returns:
            ``True`` if ``key`` exists in ``self._data``.
        """
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the Dict container.

        Returns:
            Iterator over dictionary keys.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of entries in the Dict container.

        Returns:
            Number of stored entries.
        """
        return len(self._data)


class List(Module, list[ItemT], Generic[ItemT]):
    """List container module registering ``Module`` and ``Variable`` elements in NNX state.

    Attributes:
        _pytree__has_int_keys: Internal Flax NNX pytree compatibility flag.
    """

    _pytree__has_int_keys: bool

    def __init__(
        self,
        it: Optional[Iterable[ItemT]] = None,
        /,
        *,
        _pytree__has_int_keys: bool = True,
    ) -> None:
        """Initialize a List module from an optional iterable of items.

        Args:
            it: Optional iterable of elements to populate the list.
            _pytree__has_int_keys: Pytree integer-key marker matching ``flax.nnx.List``.
        """
        Module.__init__(self)
        list.__init__(self)
        self._pytree__has_int_keys = bool(_pytree__has_int_keys)
        if it is not None:
            for item in it:
                self.append(item)
        self.finish_init()

    def _register_indexed_item(self, index: int, value: ItemT) -> None:
        """Register ``value`` under its stringified integer index if it is a ``Module`` or ``Variable``.

        Args:
            index: Non-negative integer element index.
            value: Element value.
        """
        self._ensure_registries()
        key = str(index)
        if isinstance(value, Module):
            self.register_child(key, value)
        elif isinstance(value, Variable):
            self.register_variable(key, value)

    def append(self, value: ItemT) -> None:
        """Append ``value`` to the end of the list and register it in module state.

        Args:
            value: Element to append.
        """
        idx = len(self)
        super().append(value)
        self._register_indexed_item(idx, value)

    def extend(self, values: Iterable[ItemT]) -> None:
        """Extend the list by appending all elements from ``values``.

        Args:
            values: Iterable of elements to append.
        """
        for val in values:
            self.append(val)


SequentialLayer: TypeAlias = Union[
    Module,
    LayerCallableProtocol,
    Callable[[ArrayLike], ArrayLike],
    Callable[[Union[int, float]], Union[int, float]],
    Callable[[float, float], float],
]


def _as_layer_callable(fn: SequentialLayer) -> TypeGuard[LayerCallableProtocol]:
    """Narrow a ``SequentialLayer`` to ``LayerCallableProtocol``.

    Args:
        fn: Candidate sequential layer or callable.

    Returns:
        Always ``True`` for valid sequential layers.
    """
    return callable(fn)


class Sequential(Module):
    """Sequential container module chaining a sequence of layers or callables.

    Attributes:
        layers: List of callable layers or functions executed in order.
        fns: Alias for ``layers`` for Flax NNX compatibility.
    """

    layers: list[SequentialLayer]
    fns: Tuple[SequentialLayer, ...]

    def __init__(self, *fns: SequentialLayer) -> None:
        """Initialize a Sequential container with a variable number of callables or modules.

        Args:
            *fns: Layers or functions to apply sequentially.
        """
        super().__init__()
        self.fns = tuple(fns)
        self.layers = list(fns)
        for idx, fn in enumerate(fns):
            if isinstance(fn, Module):
                self.register_child(f"layer_{idx}", fn)
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *args: ArrayLike,
        **kwargs: ArrayLike,
    ) -> ArrayLike:
        """Apply all layers in ``self.layers`` sequentially to ``x``.

        Args:
            x: Initial input array passed to the first layer.
            *args: Additional positional arguments forwarded to each layer.
            **kwargs: Additional keyword arguments forwarded to each layer.

        Returns:
            Final output produced by the last layer in the sequence.
        """
        cur: ArrayLike = x
        for fn in self.layers:
            if _as_layer_callable(fn):
                cur = fn(cur, *args, **kwargs) if (args or kwargs) else fn(cur)
        return cur


__all__ = [
    "ContainerItem",
    "Dict",
    "ItemT",
    "List",
    "Sequential",
]
