"""Predicate filter DSL for partitioning and querying NNX module state.

This module implements `Filter`, `Everything`, `Nothing`, `OfType`, `PathContains`,
`All`, `AnyFilter`, `Not`, and `to_predicate` for splitting and filtering `State`
and `Module` hierarchies without dynamic reflection or `Any`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional, Tuple, Union

from typing_extensions import TypeAlias

from zero_flax.nnx.variables import AnyVariable, VariableBase, VariableValue
from zero_flax.types import ArrayLike, JaxArray

FilterTarget: TypeAlias = Union[
    VariableBase,
    AnyVariable,
    dict[str, VariableValue],
    JaxArray,
    ArrayLike,
    int,
    float,
    str,
    bool,
    None,
]
FilterCallable: TypeAlias = Callable[[Tuple[str, ...], FilterTarget], bool]


class Filter:
    """Base predicate filter class for matching variables in an NNX state tree.

    Attributes:
        predicate: Optional custom callable taking ``(path, value)`` and returning ``bool``.
    """

    predicate: Optional[FilterCallable]

    def __init__(self, predicate: Optional[FilterCallable] = None) -> None:
        """Initialize a Filter predicate.

        Args:
            predicate: Optional custom callable evaluating ``(path, value) -> bool``.
        """
        self.predicate = predicate

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Evaluate whether ``(path, value)`` matches this filter.

        Args:
            path: Tuple of string attribute/key names leading to ``value``.
            value: Candidate ``Variable`` or state leaf value.

        Returns:
            ``True`` if the candidate matches, otherwise ``False``.
        """
        if self.predicate is not None:
            return self.predicate(path, value)
        return True

    def __and__(self, other: FilterLiteral) -> All:
        """Return the logical conjunction ``All(self, other)``.

        Args:
            other: Filter specification to combine via logical AND.

        Returns:
            Conjunction ``All`` filter.
        """
        return All(self, other)

    def __or__(self, other: FilterLiteral) -> AnyFilter:
        """Return the logical disjunction ``AnyFilter(self, other)``.

        Args:
            other: Filter specification to combine via logical OR.

        Returns:
            Disjunction ``AnyFilter`` filter.
        """
        return AnyFilter(self, other)

    def __invert__(self) -> Not:
        """Return the logical negation ``Not(self)``.

        Returns:
            Negation ``Not`` filter.
        """
        return Not(self)


class Everything(Filter):
    """Filter predicate that matches every variable in a state tree."""

    def __init__(self) -> None:
        """Initialize an Everything filter."""
        super().__init__(None)

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Return ``True`` for every ``(path, value)`` pair.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            Always ``True``.
        """
        _ = (path, value)
        return True


class Nothing(Filter):
    """Filter predicate that matches no variables in a state tree."""

    def __init__(self) -> None:
        """Initialize a Nothing filter."""
        super().__init__(None)

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Return ``False`` for every ``(path, value)`` pair.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            Always ``False``.
        """
        _ = (path, value)
        return False


class OfType(Filter):
    """Filter predicate matching variables that are instances of a target Variable class.

    Attributes:
        target_type: The ``VariableBase`` class or type to match against.
    """

    target_type: type[VariableBase]

    def __init__(self, target_type: type[VariableBase]) -> None:
        """Initialize an OfType filter.

        Args:
            target_type: The ``Variable`` subclass to match (e.g., ``Param``, ``BatchStat``).
        """
        super().__init__(None)
        self.target_type = target_type

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Check whether ``value`` is an instance of ``self.target_type``.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            ``True`` if ``isinstance(value, self.target_type)``, else ``False``.
        """
        _ = path
        return isinstance(value, self.target_type)


class PathContains(Filter):
    """Filter predicate matching variables whose path contains a target segment or substring.

    Attributes:
        key: Substring or path segment name to search for.
    """

    key: str

    def __init__(self, key: str) -> None:
        """Initialize a PathContains filter.

        Args:
            key: Substring or key segment to match against the variable path.
        """
        super().__init__(None)
        self.key = key

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Check whether ``self.key`` appears in any segment of ``path`` or slash/dot-joined path.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            ``True`` if ``self.key`` is contained in any segment or explicit slash/dot path.
        """
        _ = value
        if "/" in self.key:
            return self.key in "/".join(path)
        if "." in self.key:
            return self.key in ".".join(path)
        return any(self.key in seg for seg in path)


class All(Filter):
    """Filter predicate computing the logical conjunction (AND) of multiple filters.

    Attributes:
        filters: Tuple of child ``Filter`` predicates.
    """

    filters: Tuple[Filter, ...]

    def __init__(self, *filters: FilterLiteral) -> None:
        """Initialize an All conjunction filter.

        Args:
            *filters: Child filter specifications that must all evaluate to ``True``.
        """
        super().__init__(None)
        self.filters = tuple(to_predicate(f) for f in filters)

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Evaluate whether all child filters match ``(path, value)``.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            ``True`` if every child filter returns ``True``, else ``False``.
        """
        return all(f(path, value) for f in self.filters)


class AnyFilter(Filter):
    """Filter predicate computing the logical disjunction (OR) of multiple filters.

    Attributes:
        filters: Tuple of child ``Filter`` predicates.
    """

    filters: Tuple[Filter, ...]

    def __init__(self, *filters: FilterLiteral) -> None:
        """Initialize an AnyFilter disjunction filter.

        Args:
            *filters: Child filter specifications where at least one must evaluate to ``True``.
        """
        super().__init__(None)
        self.filters = tuple(to_predicate(f) for f in filters)

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Evaluate whether at least one child filter matches ``(path, value)``.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            ``True`` if any child filter returns ``True``, else ``False``.
        """
        return any(f(path, value) for f in self.filters)


class Not(Filter):
    """Filter predicate computing the logical negation (NOT) of an inner filter.

    Attributes:
        inner: The wrapped ``Filter`` predicate to negate.
    """

    inner: Filter

    def __init__(self, inner: FilterLiteral) -> None:
        """Initialize a Not negation filter.

        Args:
            inner: Filter specification whose boolean result will be inverted.
        """
        super().__init__(None)
        self.inner = to_predicate(inner)

    def __call__(self, path: Tuple[str, ...], value: FilterTarget) -> bool:
        """Return the logical negation of ``self.inner(path, value)``.

        Args:
            path: Tuple of string path segments.
            value: Candidate variable or state entry.

        Returns:
            ``True`` if ``self.inner`` returns ``False``, else ``False``.
        """
        return not self.inner(path, value)


FilterLiteral: TypeAlias = Union[
    Filter,
    type[VariableBase],
    FilterCallable,
    Sequence["FilterLiteral"],
    str,
    bool,
    None,
]


def to_predicate(filter_spec: FilterLiteral) -> Filter:
    """Convert a filter specification literal into a concrete ``Filter`` instance.

    Args:
        filter_spec: A ``Filter`` object, ``Variable`` subclass, callable predicate,
            sequence of filter literals, ``Ellipsis`` (``...``), path substring ``str``,
            ``bool``, or ``None``.

    Returns:
        Concrete ``Filter`` predicate object.
    """
    if isinstance(filter_spec, Filter):
        return filter_spec
    if filter_spec is True:
        return Everything()
    if filter_spec is False or filter_spec is None:
        return Nothing()
    if isinstance(filter_spec, str):
        return PathContains(filter_spec)
    if isinstance(filter_spec, Sequence):
        return AnyFilter(*filter_spec)
    if isinstance(filter_spec, type):
        return OfType(filter_spec)
    return Filter(filter_spec)


__all__ = [
    "All",
    "AnyFilter",
    "Everything",
    "Filter",
    "FilterCallable",
    "FilterLiteral",
    "FilterTarget",
    "Not",
    "Nothing",
    "OfType",
    "PathContains",
    "to_predicate",
]
