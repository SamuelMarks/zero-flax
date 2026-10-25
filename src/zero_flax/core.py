"""Core Flax NNX and `flax.core` compatibility exports for `zero_flax.core`.

This module provides `FrozenDict`, `freeze`, `unfreeze`, `copy`, `pop`, and
`pretty_repr` alongside the strongly-typed `Module`, `GraphDef`, `State`,
`Variable` subclasses, layers, containers, and transformations from `zero_flax.nnx`
without duplicate stub classes or `object` annotations.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Tuple, TypeVar, Union

from typing_extensions import TypeAlias

from zero_flax.nnx import (
    RNN,
    BatchNorm,
    BatchStat,
    Bidirectional,
    Cache,
    Conv,
    ConvTranspose,
    Dense,
    Dict,
    Dropout,
    Einsum,
    Embed,
    GraphDef,
    GroupNorm,
    GRUCell,
    InstanceNorm,
    Intermediate,
    Jit,
    LayerNorm,
    Linear,
    LinearGeneral,
    List,
    LoRA,
    LoRALinear,
    LoRAParam,
    LSTMCell,
    Module,
    MultiHeadAttention,
    MultiHeadDotProductAttention,
    OptimizedLSTMCell,
    Optimizer,
    OptState,
    OptVariable,
    Param,
    Pmap,
    PReLU,
    Remat,
    RMSNorm,
    Rng,
    RngCount,
    RngKey,
    Rngs,
    RngState,
    RngStream,
    RNNCellBase,
    RoPE,
    Scan,
    Sequential,
    SimpleCell,
    SpectralNorm,
    State,
    Variable,
    Vmap,
    WeightNorm,
    iter_graph,
    merge,
    restore_state,
    save_state,
    split,
    update,
)
from zero_flax.nnx.module import state as state
from zero_flax.nnx.variables import AnyVariable, VariableBase, VariableValue

K = TypeVar("K", bound=str)
V = TypeVar("V")
FrozenEntry: TypeAlias = Union[
    VariableValue,
    VariableBase,
    AnyVariable,
    "FrozenDict",
    Mapping[str, "FrozenEntry"],
]


def _freeze_entry(val: FrozenEntry) -> FrozenEntry:
    """Recursively convert any mapping ``val`` into a ``FrozenDict``.

    Args:
        val: Candidate entry value.

    Returns:
        Frozen ``FrozenDict`` if ``val`` is a mapping, otherwise ``val`` unchanged.
    """
    if isinstance(val, FrozenDict):
        return val
    if isinstance(val, Mapping):
        return FrozenDict(val)
    return val


class FrozenDict(Mapping[str, FrozenEntry]):
    """Immutable hierarchical dictionary compatible with ``flax.core.FrozenDict``.

    Attributes:
        _data: Internal read-only dictionary mapping string keys to entries.
    """

    _data: dict[str, FrozenEntry]

    def __init__(
        self,
        mapping_or_iterable: Union[
            Mapping[str, FrozenEntry],
            Iterable[Tuple[str, FrozenEntry]],
            None,
        ] = None,
        **kwargs: FrozenEntry,
    ) -> None:
        """Initialize a FrozenDict from an optional mapping/iterable and keyword entries.

        Args:
            mapping_or_iterable: Optional source mapping or key-value iterable.
            **kwargs: Additional key-value entries.
        """
        raw: dict[str, FrozenEntry] = {}
        if mapping_or_iterable is not None:
            if isinstance(mapping_or_iterable, Mapping):
                for k, v in mapping_or_iterable.items():
                    raw[k] = _freeze_entry(v)
            else:
                for k, v in mapping_or_iterable:
                    raw[k] = _freeze_entry(v)
        for k, v in kwargs.items():
            raw[k] = _freeze_entry(v)
        self._data = raw

    def __getitem__(self, key: str) -> FrozenEntry:
        """Retrieve the entry associated with ``key``.

        Args:
            key: Dictionary key.

        Returns:
            Associated ``FrozenEntry``.
        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys of the FrozenDict.

        Returns:
            Key iterator.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of top-level entries in the FrozenDict.

        Returns:
            Number of entries.
        """
        return len(self._data)

    def __hash__(self) -> int:
        """Compute a deterministic hash of the FrozenDict contents.

        Returns:
            Integer hash value.
        """
        items_hash: list[Tuple[str, int]] = []
        for k in sorted(self._data.keys()):
            v = self._data[k]
            v_hash = (
                hash(v) if isinstance(v, (int, float, str, bool, FrozenDict)) else hash(repr(v))
            )
            items_hash.append((k, v_hash))
        return hash(tuple(items_hash))

    def copy(
        self,
        add_or_replace: Union[Mapping[str, FrozenEntry], None] = None,
    ) -> FrozenDict:
        """Return a new FrozenDict with optional updated or added entries.

        Args:
            add_or_replace: Optional mapping of keys to add or replace.

        Returns:
            New ``FrozenDict`` instance.
        """
        merged = dict(self._data)
        if add_or_replace is not None:
            for k, v in add_or_replace.items():
                merged[k] = _freeze_entry(v)
        return FrozenDict(merged)

    def pop(self, key: str) -> Tuple[FrozenDict, FrozenEntry]:
        """Return a new FrozenDict without ``key`` and the removed value.

        Args:
            key: Key to remove.

        Returns:
            Pair ``(new_frozen_dict, popped_value)``.
        """
        new_data = dict(self._data)
        val = new_data.pop(key)
        return FrozenDict(new_data), val

    def unfreeze(self) -> dict[str, FrozenEntry]:
        """Recursively convert this FrozenDict into a mutable Python ``dict``.

        Returns:
            Mutable nested ``dict``.
        """
        out: dict[str, FrozenEntry] = {}
        for k, v in self._data.items():
            out[k] = v.unfreeze() if isinstance(v, FrozenDict) else v
        return out

    def __repr__(self) -> str:
        """Return a formatted string representation of the FrozenDict.

        Returns:
            Formatted ``"FrozenDict(...)"`` string.
        """
        return f"FrozenDict({self._data!r})"


def freeze(xs: Mapping[str, FrozenEntry]) -> FrozenDict:
    """Freeze a mutable mapping into an immutable ``FrozenDict``.

    Args:
        xs: Mapping to freeze.

    Returns:
        Immutable ``FrozenDict`` instance.
    """
    if isinstance(xs, FrozenDict):
        return xs
    return FrozenDict(xs)


def unfreeze(xs: Union[FrozenDict, Mapping[str, FrozenEntry]]) -> dict[str, FrozenEntry]:
    """Unfreeze a ``FrozenDict`` or mapping into a mutable Python ``dict``.

    Args:
        xs: ``FrozenDict`` or mapping to unfreeze.

    Returns:
        Mutable ``dict`` copy.
    """
    if isinstance(xs, FrozenDict):
        return xs.unfreeze()
    return {k: unfreeze(v) if isinstance(v, Mapping) else v for k, v in xs.items()}


def copy(
    xs: Union[FrozenDict, Mapping[str, FrozenEntry]],
    add_or_replace: Union[Mapping[str, FrozenEntry], None] = None,
) -> FrozenDict:
    """Copy a ``FrozenDict`` or mapping, applying optional ``add_or_replace`` updates.

    Args:
        xs: Source ``FrozenDict`` or mapping.
        add_or_replace: Optional key-value updates.

    Returns:
        Updated ``FrozenDict``.
    """
    return freeze(xs).copy(add_or_replace)


def pop(
    xs: Union[FrozenDict, Mapping[str, FrozenEntry]],
    key: str,
) -> Tuple[FrozenDict, FrozenEntry]:
    """Remove ``key`` from ``xs`` and return ``(remaining_frozen_dict, value)``.

    Args:
        xs: Source ``FrozenDict`` or mapping.
        key: Key to pop.

    Returns:
        Pair ``(remaining_frozen_dict, popped_value)``.
    """
    return freeze(xs).pop(key)


def pretty_repr(xs: Union[FrozenDict, Mapping[str, FrozenEntry]], num_spaces: int = 4) -> str:
    """Return an indented human-readable string representation of ``xs``.

    Args:
        xs: Mapping or ``FrozenDict`` to format.
        num_spaces: Indentation width in spaces. Defaults to ``4``.

    Returns:
        Indented string representation.
    """
    nl = chr(10)
    indent = " " * max(0, int(num_spaces))
    lines: list[str] = ["{"]
    for k, v in xs.items():
        if isinstance(v, Mapping):
            sub = pretty_repr(v, num_spaces=num_spaces).replace(nl, f"{nl}{indent}")
            lines.append(f"{indent}{k}: {sub},")
        else:
            lines.append(f"{indent}{k}: {v!r},")
    lines.append("}")
    return nl.join(lines)


__all__ = [
    "BatchNorm",
    "BatchStat",
    "Bidirectional",
    "Cache",
    "Conv",
    "ConvTranspose",
    "Dense",
    "Dict",
    "Dropout",
    "Einsum",
    "Embed",
    "FrozenDict",
    "FrozenEntry",
    "GRUCell",
    "GraphDef",
    "GroupNorm",
    "InstanceNorm",
    "Intermediate",
    "Jit",
    "LSTMCell",
    "LayerNorm",
    "Linear",
    "LinearGeneral",
    "List",
    "LoRA",
    "LoRALinear",
    "LoRAParam",
    "Module",
    "MultiHeadAttention",
    "MultiHeadDotProductAttention",
    "OptState",
    "OptVariable",
    "OptimizedLSTMCell",
    "Optimizer",
    "PReLU",
    "Param",
    "Pmap",
    "RMSNorm",
    "RNN",
    "RNNCellBase",
    "Remat",
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "RngStream",
    "Rngs",
    "RoPE",
    "Scan",
    "Sequential",
    "SimpleCell",
    "SpectralNorm",
    "State",
    "Variable",
    "Vmap",
    "WeightNorm",
    "copy",
    "freeze",
    "iter_graph",
    "merge",
    "pop",
    "pretty_repr",
    "restore_state",
    "save_state",
    "split",
    "state",
    "unfreeze",
    "update",
]
