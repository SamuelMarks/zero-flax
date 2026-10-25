"""Container modules (`Dict`, `List`, `Sequential`) for the zero-flax NNX API.

This module implements standard container modules that inherit from `Module`
and automatically register child `Module` and `Variable` entries into the
parent module registries without dynamic reflection or `Any`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import (
    Generic,
    Optional,
    Protocol,
    SupportsIndex,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from typing_extensions import TypeAlias, TypeGuard

from zero_flax.nnx.module import Module
from zero_flax.nnx.variables import AnyVariable, Variable, VariableBase
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
    VariableBase,
    AnyVariable,
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
        self._allows_dynamic_variables = True
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
        """Store ``value`` under ``key`` and synchronize child/variable registries.

        Args:
            key: String dictionary key.
            value: Item to store.
        """
        self._ensure_registries()
        str_key = str(key)
        self._children.pop(str_key, None)
        self._variables.pop(str_key, None)
        self._data[str_key] = value
        if isinstance(value, Module):
            self.register_child(str_key, value)
        elif isinstance(value, Variable):
            self.register_variable(str_key, value)

    def __getitem__(self, key: str) -> ItemT:
        """Retrieve the item stored under ``key``.

        Args:
            key: String dictionary key.

        Returns:
            Stored item.
        """
        return self._data[str(key)]

    def __delitem__(self, key: str) -> None:
        """Delete ``key`` from the container and remove it from child/variable registries.

        Args:
            key: String dictionary key to delete.
        """
        str_key = str(key)
        del self._data[str_key]
        self._children.pop(str_key, None)
        self._variables.pop(str_key, None)

    def get(self, key: str, default: Optional[ItemT] = None) -> Optional[ItemT]:
        """Return the item for ``key`` if present, else ``default``.

        Args:
            key: String dictionary key.
            default: Fallback value if ``key`` is absent.

        Returns:
            Stored item or ``default``.
        """
        if key in self._data:
            return self[key]
        return default

    def pop(self, key: str, default: Optional[ItemT] = None) -> Optional[ItemT]:
        """Remove ``key`` and return its value, or ``default`` if absent.

        Args:
            key: String dictionary key to pop.
            default: Fallback value if ``key`` is absent.

        Returns:
            Removed item or ``default``.
        """
        str_key = str(key)
        if str_key in self._data:
            val = self[str_key]
            del self[str_key]
            return val
        return default

    def clear(self) -> None:
        """Remove all items from the Dict container and clear child/variable registries."""
        self._data.clear()
        self._children.clear()
        self._variables.clear()

    def keys(self) -> list[str]:
        """Return a list of keys in the Dict container.

        Returns:
            List of string keys.
        """
        return list(self._data.keys())

    def values(self) -> list[ItemT]:
        """Return a list of values in the Dict container.

        Returns:
            List of stored items.
        """
        return [self[k] for k in self._data]

    def items(self) -> list[Tuple[str, ItemT]]:
        """Return a list of ``(key, value)`` pairs in the Dict container.

        Returns:
            List of ``(key, value)`` tuples.
        """
        return [(k, self[k]) for k in self._data]

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
        _pytree__has_int_keys: bool = True,
    ) -> None:
        """Initialize a List module from an optional iterable of items.

        Args:
            it: Optional iterable of elements to populate the list.
            _pytree__has_int_keys: Pytree integer-key marker matching ``flax.nnx.List``.
        """
        Module.__init__(self)
        list.__init__(self)
        self._allows_dynamic_variables = True
        self._pytree__has_int_keys = bool(_pytree__has_int_keys)
        if it is not None:
            for item in it:
                self.append(item)
        self.finish_init()

    def _rebuild_registries(self) -> None:
        """Rebuild ``_children`` and ``_variables`` from the current list contents in index order."""
        self._ensure_registries()
        self._children.clear()
        self._variables.clear()
        for idx, item in enumerate(list.__iter__(self)):
            key = str(idx)
            if isinstance(item, Module):
                self.register_child(key, item)
            elif isinstance(item, Variable):
                self.register_variable(key, item)

    def append(self, value: ItemT) -> None:
        """Append ``value`` to the end of the list and synchronize module state.

        Args:
            value: Element to append.
        """
        super().append(value)
        self._rebuild_registries()

    def extend(self, values: Iterable[ItemT]) -> None:
        """Extend the list by appending all elements from ``values``.

        Args:
            values: Iterable of elements to append.
        """
        super().extend(values)
        self._rebuild_registries()

    def insert(self, index: SupportsIndex, value: ItemT) -> None:
        """Insert ``value`` before ``index`` and re-index registered children and variables.

        Args:
            index: Insertion position index.
            value: Element to insert.
        """
        super().insert(index, value)
        self._rebuild_registries()

    def pop(self, index: SupportsIndex = -1) -> ItemT:
        """Remove and return item at ``index`` and re-index registered children and variables.

        Args:
            index: Index of element to remove. Defaults to ``-1``.

        Returns:
            Removed element.
        """
        removed = super().pop(index)
        self._rebuild_registries()
        return removed

    def clear(self) -> None:
        """Remove all elements from the list and clear child/variable registries."""
        super().clear()
        self._rebuild_registries()

    def set_at(self, index: int, value: ItemT) -> None:
        """Replace the element at ``index`` with ``value`` and synchronize registries.

        Args:
            index: Element index to replace.
            value: New element value.
        """
        list.__setitem__(self, index, value)
        self._rebuild_registries()

    def delete_at(self, index: int) -> None:
        """Delete the element at ``index`` and re-index registries.

        Args:
            index: Element index to delete.
        """
        list.__delitem__(self, index)
        self._rebuild_registries()


SequentialLayer: TypeAlias = Union[
    Module,
    LayerCallableProtocol,
    Callable[[ArrayLike], ArrayLike],
    Callable[[Union[int, float]], Union[int, float]],
    Callable[[float, float], float],
]


def _as_layer_callable(fn: SequentialLayer) -> TypeGuard[LayerCallableProtocol]:
    """Narrow a ``SequentialLayer`` to ``LayerCallableProtocol`` if callable.

    Args:
        fn: Candidate sequential layer or callable.

    Returns:
        ``True`` if ``fn`` is callable, otherwise ``False``.
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

        Raises:
            TypeError: If any layer in ``self.layers`` is not callable.
        """
        cur: ArrayLike = x
        for fn in self.layers:
            if not _as_layer_callable(fn):
                raise TypeError(f"Sequential layer {fn!r} is not callable.")
            cur = fn(cur, *args, **kwargs) if (args or kwargs) else fn(cur)
        return cur


__all__ = [
    "ContainerItem",
    "Dict",
    "ItemT",
    "List",
    "Sequential",
]
