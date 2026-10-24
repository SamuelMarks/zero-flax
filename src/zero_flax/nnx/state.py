"""Hierarchical State dictionary and Variable exports for the zero-flax NNX API.

This module implements `State` and `merge` for managing nested dictionaries of
NNX `Variable` instances (`Param`, `BatchStat`, `Cache`, `Intermediate`, `Rng`,
and `LoRAParam`) without `Any`, `object`, or reflection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Tuple, Union

from typing_extensions import TypeAlias

from zero_flax.nnx.filterlib import Filter, FilterLiteral, to_predicate
from zero_flax.nnx.variables import (
    BatchStat,
    Cache,
    Intermediate,
    LoRAParam,
    Param,
    Rng,
    RngCount,
    RngKey,
    RngState,
    V,
    Variable,
    VariableValue,
)

StateEntry: TypeAlias = Union[
    Variable[VariableValue],
    "State",
    VariableValue,
]

FlatPathKey: TypeAlias = Union[str, Tuple[str, ...]]


class State(dict[str, StateEntry]):
    """Dictionary-like container representing the hierarchical state of an NNX Module.

    Attributes:
        (Inherits mapping storage from ``dict[str, StateEntry]``.)
    """

    def __init__(
        self,
        mapping_or_iterable: Union[
            Mapping[str, StateEntry],
            Iterable[Tuple[str, StateEntry]],
            None,
        ] = None,
        **kwargs: StateEntry,
    ) -> None:
        """Initialize a State container from an optional mapping/iterable and keyword entries.

        Args:
            mapping_or_iterable: Optional initial mapping or iterable of ``(key, value)`` pairs.
            **kwargs: Additional key-value state entries.
        """
        if mapping_or_iterable is not None:
            super().__init__(mapping_or_iterable)
        else:
            super().__init__()
        if kwargs:
            self.update(kwargs)

    def _partition_with_predicates(
        self,
        predicates: Tuple[Filter, ...],
        prefix: Tuple[str, ...],
    ) -> Tuple[list[State], State]:
        """Recursively partition this State across ``predicates`` with path tracking.

        Args:
            predicates: Tuple of compiled ``Filter`` predicates.
            prefix: Tuple of parent path keys leading to this ``State``.

        Returns:
            Pair ``(matched_states_list, rest_state)``.
        """
        matched_states: list[State] = [State() for _ in predicates]
        rest = State()
        for key, val in self.items():
            current_path = prefix + (key,)
            if isinstance(val, State):
                sub_matches, sub_rest = val._partition_with_predicates(predicates, current_path)
                for idx, sub_m in enumerate(sub_matches):
                    if len(sub_m) > 0:
                        matched_states[idx][key] = sub_m
                if len(sub_rest) > 0:
                    rest[key] = sub_rest
            else:
                placed = False
                filter_val: Union[Variable[VariableValue], int, float, str, bool, None] = (
                    val if isinstance(val, (Variable, int, float, str, bool)) else None
                )
                for idx, pred in enumerate(predicates):
                    if pred(current_path, filter_val):
                        matched_states[idx][key] = val
                        placed = True
                        break
                if not placed:
                    rest[key] = val
        return matched_states, rest

    def split(self, *filters: FilterLiteral) -> Tuple[State, ...]:
        """Partition the State into multiple ``State`` objects according to ``filters``.

        Each entry is placed into the first matching filter bucket; any unmatched
        entries are placed into the trailing ``rest`` bucket.

        Args:
            *filters: Variable types (e.g., ``Param``, ``BatchStat``) or ``Filter`` predicates.

        Returns:
            A tuple ``(*matched_states, rest_state)`` of ``State`` instances.
        """
        predicates = tuple(to_predicate(f) for f in filters)
        matched_states, rest = self._partition_with_predicates(predicates, ())
        return tuple(matched_states) + (rest,)

    def flat_state(self) -> dict[str, Variable[VariableValue]]:
        """Flatten nested ``State`` dictionaries into a single dot-separated path mapping.

        Returns:
            Dictionary mapping dot-joined paths (e.g., ``"d1.kernel"``) to ``Variable`` leaves.
        """
        flat: dict[str, Variable[VariableValue]] = {}
        for key, val in self.items():
            if isinstance(val, State):
                for sub_key, sub_val in val.flat_state().items():
                    flat[f"{key}.{sub_key}"] = sub_val
            elif isinstance(val, Variable):
                flat[key] = val
        return flat

    def flat_tuple_state(self) -> dict[Tuple[str, ...], Variable[VariableValue]]:
        """Flatten nested ``State`` dictionaries into a mapping keyed by tuple path segments.

        Returns:
            Dictionary mapping tuple paths (e.g., ``("d1", "kernel")``) to ``Variable`` leaves.
        """
        flat: dict[Tuple[str, ...], Variable[VariableValue]] = {}
        for key, val in self.items():
            if isinstance(val, State):
                for sub_path, sub_val in val.flat_tuple_state().items():
                    flat[(key, *sub_path)] = sub_val
            elif isinstance(val, Variable):
                flat[(key,)] = val
        return flat

    @classmethod
    def from_flat_state(
        cls,
        flat: Union[
            Mapping[str, Variable[VariableValue]],
            Mapping[Tuple[str, ...], Variable[VariableValue]],
            Mapping[FlatPathKey, Variable[VariableValue]],
        ],
    ) -> State:
        """Reconstruct a hierarchical ``State`` from a dot-separated or tuple-path dictionary.

        Args:
            flat: Mapping from dot-separated string keys or tuple-of-strings keys to ``Variable`` instances.

        Returns:
            Reconstructed hierarchical ``State``.
        """
        root = cls()
        for raw_key, var in flat.items():
            parts = tuple(raw_key.split(".")) if isinstance(raw_key, str) else tuple(raw_key)
            cursor = root
            for segment in parts[:-1]:
                existing = cursor.get(segment)
                if isinstance(existing, State):
                    cursor = existing
                else:
                    child = cls()
                    cursor[segment] = child
                    cursor = child
            cursor[parts[-1]] = var
        return root


def merge(*states: State) -> State:
    """Merge multiple ``State`` objects recursively into a single ``State``.

    Later states override keys from earlier states; nested ``State`` dictionaries
    under the same key are merged recursively.

    Args:
        *states: One or more ``State`` dictionaries to merge.

    Returns:
        Newly merged ``State`` container.
    """
    result = State()
    for state in states:
        for key, val in state.items():
            existing = result.get(key)
            if isinstance(existing, State) and isinstance(val, State):
                result[key] = merge(existing, val)
            else:
                result[key] = val
    return result


__all__ = [
    "BatchStat",
    "Cache",
    "FlatPathKey",
    "Intermediate",
    "LoRAParam",
    "Param",
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "State",
    "StateEntry",
    "V",
    "Variable",
    "VariableValue",
    "merge",
]
