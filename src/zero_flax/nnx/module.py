"""Reflection-free base Module, GraphDef, split, and merge system for zero-flax NNX.

This module implements `Module`, `GraphDef`, `split`, `merge`, `update`, `state`,
`clone`, and `iter_graph` using explicit typed registries (`_variables`, `_children`,
`_static_config`) and deep blueprint cloning (`copy.deepcopy`), completely avoiding
`getattr`, `setattr`, `hasattr`, `vars`, `__setattr__`, `__dict__`, `__new__`, and `Any`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from typing import Generic, Optional, Tuple, TypeVar, Union, overload

from zero_flax.nnx.filterlib import FilterLiteral
from zero_flax.nnx.state import State, StateEntry, _state_to_chex_pytree
from zero_flax.nnx.state import merge as state_merge
from zero_flax.nnx.variables import AnyVariable, Variable
from zero_flax.types import MetadataValue

M = TypeVar("M", bound="Module")
ChildT = TypeVar("ChildT", bound="Module")
VarT = TypeVar("VarT", bound=AnyVariable)


def _filter_state_to_keys(reference: State, subset: State) -> State:
    """Extract the subset of ``reference`` matching the keys present in ``subset``.

    Args:
        reference: Full reference ``State`` tree.
        subset: Partial or complete ``State`` tree whose keys select entries from ``reference``.

    Returns:
        Filtered ``State`` containing only keys present in both ``reference`` and ``subset``.
    """
    filtered = State()
    for key, sub_val in subset.items():
        if key in reference:
            ref_val = reference[key]
            if isinstance(ref_val, State) and isinstance(sub_val, State):
                filtered[key] = _filter_state_to_keys(ref_val, sub_val)
            else:
                filtered[key] = ref_val
    return filtered


class Module:
    """Base class for all neural network modules in zero-flax NNX.

    Subclasses register parameters/variables via ``register_variable`` and child
    sub-modules via ``register_child``, enabling reflection-free state extraction,
    updates, splitting, and merging.

    Attributes:
        _registries_initialized: Class-level sentinel flag overridden on instance creation.
        _children: Dictionary mapping child attribute names to child ``Module`` instances.
        _variables: Dictionary mapping variable names to ``Variable`` instances.
        _static_config: Dictionary mapping static configuration keys to metadata values.
        _is_initializing: Boolean flag indicating whether ``__init__`` is in progress.
        _allows_dynamic_variables: Boolean flag indicating whether ``update()`` may insert
            new variable keys (enabled on dynamic containers such as ``Dict`` and ``List``).
    """

    _registries_initialized: bool = False
    _children: dict[str, Module]
    _variables: dict[str, AnyVariable]
    _static_config: dict[str, MetadataValue]
    _is_initializing: bool
    _allows_dynamic_variables: bool

    def __init__(self, **_kwargs: MetadataValue) -> None:
        """Initialize the base Module registries.

        Args:
            **_kwargs: Optional static configuration metadata entries.
        """
        self._children = {}
        self._variables = {}
        self._static_config = dict(_kwargs)
        self._is_initializing = True
        self._allows_dynamic_variables = False
        self._registries_initialized = True

    def _ensure_registries(self) -> None:
        """Ensure internal registries exist even if ``super().__init__()`` was not called first."""
        if not self._registries_initialized:
            self._children = {}
            self._variables = {}
            self._static_config = {}
            self._is_initializing = False
            self._allows_dynamic_variables = False
            self._registries_initialized = True

    def register_variable(self, name: str, variable: VarT) -> VarT:
        """Register a stateful ``Variable`` (such as ``Param`` or ``BatchStat``) under ``name``.

        Args:
            name: Attribute/key name of the variable within this module.
            variable: The ``Variable`` instance to register.

        Returns:
            The registered ``variable`` instance.
        """
        self._ensure_registries()
        self._variables[name] = variable
        return variable

    def register_child(self, name: str, child: ChildT) -> ChildT:
        """Register a child ``Module`` under ``name``.

        Args:
            name: Attribute/key name of the child module.
            child: The child ``Module`` instance to register.

        Returns:
            The registered ``child`` module.
        """
        self._ensure_registries()
        self._children[name] = child
        return child

    def register_static(self, name: str, value: MetadataValue) -> None:
        """Register a static configuration value under ``name``.

        Args:
            name: Configuration key name.
            value: Static metadata value.
        """
        self._ensure_registries()
        self._static_config[name] = value

    def finish_init(self) -> None:
        """Mark module initialization as finished."""
        self._ensure_registries()
        self._is_initializing = False

    def is_initializing(self) -> bool:
        """Return whether this module is still executing ``__init__``.

        Returns:
            ``True`` before ``finish_init()`` is called, ``False`` afterward.
        """
        self._ensure_registries()
        return self._is_initializing

    def _on_state_updated(self) -> None:
        """Synchronize container-specific storage after ``update()`` mutates registries."""

    def state(self) -> State:
        """Extract the hierarchical ``State`` of this module and all child modules.

        Returns:
            A ``State`` dictionary containing all variables and nested child states.
        """
        self._ensure_registries()
        result = State()
        for name, var in self._variables.items():
            result[name] = var
        for name, child in self._children.items():
            result[name] = child.state()
        return result

    def update(
        self,
        state: Optional[Union[State, Sequence[State]]] = None,
        *more_states: State,
        **kw_states: StateEntry,
    ) -> None:
        """Update this module's variables and child modules in-place from one or more ``State`` objects.

        Args:
            state: Primary ``State`` dictionary or sequence of ``State`` dictionaries.
            *more_states: Additional ``State`` dictionaries to merge and apply.
            **kw_states: Additional key-value state entries to apply (supports ``strict_shapes=True``).

        Raises:
            KeyError: If a nested ``State`` key does not correspond to a registered child ``Module``,
                or if a variable/static key is unknown on a fixed-schema ``Module``.
        """
        self._ensure_registries()
        combined = State()
        if isinstance(state, State):
            combined = state_merge(combined, state)
        elif isinstance(state, Sequence):
            combined = state_merge(combined, *state)
        for extra in more_states:
            combined = state_merge(combined, extra)
        if kw_states:
            strict_flag = kw_states.pop("strict_shapes", False)
            if kw_states:
                combined = state_merge(combined, State(kw_states))
            if strict_flag is True:
                from zero_chex import assert_trees_all_equal_shapes

                ref_subset = _filter_state_to_keys(self.state(), combined)
                assert_trees_all_equal_shapes(
                    _state_to_chex_pytree(ref_subset),
                    _state_to_chex_pytree(combined),
                )

        for key, val in combined.items():
            if isinstance(val, State):
                if key not in self._children:
                    raise KeyError(f"Unknown child module key {key!r} in State update.")
                self._children[key].update(val)
            elif isinstance(val, Variable):
                if key in self._variables:
                    self._variables[key].value = val.value
                    self._variables[key].metadata = dict(val.metadata)
                elif self._allows_dynamic_variables:
                    self._variables[key] = val.copy()
                else:
                    raise KeyError(
                        f"Unknown variable key {key!r} in State update for {type(self).__name__}."
                    )
            else:
                if key in self._variables:
                    self._variables[key].value = val
                elif key in self._static_config or self._allows_dynamic_variables:
                    if isinstance(val, (str, int, float, bool, tuple)) or val is None:
                        self._static_config[key] = val
                    else:
                        raise KeyError(
                            f"Unsupported static config value for key {key!r} in {type(self).__name__}."
                        )
                else:
                    raise KeyError(
                        f"Unknown state key {key!r} in State update for {type(self).__name__}."
                    )
        self._on_state_updated()

    def _clone_blueprint(self: M) -> M:
        """Create an independent deep clone of this module preserving registry identity links.

        Returns:
            Cloned module instance of the same concrete type ``M``.
        """
        self._ensure_registries()
        cloned: M = copy.deepcopy(self)
        cloned._is_initializing = False
        return cloned


class GraphDef(Generic[M]):
    """Static graph definition of a ``Module`` produced by ``nnx.split``.

    ``GraphDef`` holds the structural blueprint of the module and its children so
    ``GraphDef.merge`` (or ``nnx.merge``) can reconstruct a live ``Module`` with
    updated ``State`` without reflection or ``__new__`` hacks.

    Attributes:
        cls: Concrete ``Module`` class represented by this graph definition.
        children_defs: Dictionary mapping child names to their ``GraphDef`` definitions.
        static_fields: Dictionary of static configuration fields.
        variable_keys: Tuple of variable keys expected directly on this module node.
        _blueprint: Cloned blueprint instance used for deterministic reconstruction.
    """

    cls: type[M]
    children_defs: dict[str, GraphDef[Module]]
    static_fields: dict[str, MetadataValue]
    variable_keys: Tuple[str, ...]
    _blueprint: M

    def __init__(
        self,
        cls: type[M],
        children_defs: dict[str, GraphDef[Module]],
        static_fields: dict[str, MetadataValue],
        blueprint: M,
        variable_keys: Tuple[str, ...] = (),
    ) -> None:
        """Initialize a GraphDef instance.

        Args:
            cls: The ``Module`` class type.
            children_defs: Mapping of child module names to their ``GraphDef`` instances.
            static_fields: Static configuration fields of the module.
            blueprint: Cloned module blueprint used during ``merge``.
            variable_keys: Optional tuple of direct variable names registered on this module.
        """
        self.cls = cls
        self.children_defs = dict(children_defs)
        self.static_fields = dict(static_fields)
        self.variable_keys = (
            tuple(variable_keys) if variable_keys else tuple(blueprint._variables.keys())
        )
        self._blueprint = blueprint

    def _apply_graphdef_metadata(self, target: Module) -> None:
        """Recursively synchronize ``static_fields`` from ``self`` and ``children_defs`` onto ``target``.

        Args:
            target: Reconstructed ``Module`` instance.
        """
        target._static_config.update(self.static_fields)
        for child_name, child_def in self.children_defs.items():
            child_def._apply_graphdef_metadata(target._children[child_name])
        target._on_state_updated()

    def merge(
        self,
        state: Union[State, Sequence[State]],
        *more_states: State,
    ) -> M:
        """Reconstruct a new ``Module`` instance from this ``GraphDef`` and the given ``State``(s).

        Args:
            state: Primary ``State`` or sequence of ``State`` objects.
            *more_states: Additional ``State`` objects to merge before applying.

        Returns:
            Reconstructed ``Module`` instance of type ``M`` populated with ``state``.
        """
        reconstructed: M = self._blueprint._clone_blueprint()
        self._apply_graphdef_metadata(reconstructed)
        if isinstance(state, State):
            combined = state_merge(state, *more_states)
        else:
            combined = state_merge(*state, *more_states)
        reconstructed.update(combined)
        return reconstructed


@overload
def split(
    module: M,
) -> Tuple[GraphDef[M], State]:
    """Split ``module`` into ``(GraphDef, State)``.

    Args:
        module: The ``Module`` instance to split.

    Returns:
        Pair ``(graphdef, state)``.
    """


@overload
def split(
    module: M,
    f1: FilterLiteral,
    f2: FilterLiteral,
    /,
) -> Tuple[GraphDef[M], State, State, State]:
    """Split ``module`` with two filters into ``(GraphDef, state_1, state_2, rest_state)``.

    Args:
        module: The ``Module`` instance to split.
        f1: First variable filter predicate.
        f2: Second variable filter predicate.

    Returns:
        Tuple ``(graphdef, state_1, state_2, rest_state)``.
    """


@overload
def split(
    module: M,
    *filters: FilterLiteral,
) -> Tuple[Union[GraphDef[M], State], ...]:
    """Split ``module`` with arbitrary filters into ``(GraphDef, *filtered_states, rest_state)``.

    Args:
        module: The ``Module`` instance to split.
        *filters: Variable filter predicates.

    Returns:
        Tuple ``(graphdef, *matched_states, rest_state)``.
    """


def split(
    module: M,
    *filters: FilterLiteral,
) -> Union[
    Tuple[GraphDef[M], State],
    Tuple[GraphDef[M], State, State, State],
    Tuple[Union[GraphDef[M], State], ...],
]:
    """Split a ``Module`` into its static ``GraphDef`` and one or more ``State`` partitions.

    Args:
        module: The ``Module`` instance to split.
        *filters: Optional variable filters (e.g., ``Param``, ``BatchStat``) used to partition state.

    Returns:
        ``(graphdef, state)`` when no filters are given, or ``(graphdef, *matched_states, rest_state)``
        when filters are provided.
    """
    module._ensure_registries()
    children_defs: dict[str, GraphDef[Module]] = {}
    for name, child in module._children.items():
        child_gdef, _ = split(child)
        children_defs[name] = child_gdef

    blueprint = module._clone_blueprint()
    graphdef = GraphDef(
        cls=type(module),
        children_defs=children_defs,
        static_fields=dict(module._static_config),
        blueprint=blueprint,
        variable_keys=tuple(module._variables.keys()),
    )
    full_state = module.state()
    if not filters:
        return graphdef, full_state
    partitioned = full_state.split(*filters)
    return (graphdef, *partitioned)


@overload
def merge(
    first: GraphDef[M],
    *rest: Union[State, Sequence[State]],
) -> M:
    """Reconstruct a ``Module`` from ``first`` (``GraphDef``) and ``*rest`` states.

    Args:
        first: ``GraphDef`` blueprint.
        *rest: ``State`` dictionaries to merge into the module.

    Returns:
        Reconstructed ``Module`` instance.
    """


@overload
def merge(
    first: State,
    *rest: Union[State, Sequence[State]],
) -> State:
    """Merge multiple ``State`` dictionaries into a single ``State``.

    Args:
        first: Initial ``State`` dictionary.
        *rest: Additional ``State`` dictionaries to merge.

    Returns:
        Merged ``State`` dictionary.
    """


def merge(
    first: Union[GraphDef[M], State],
    *rest: Union[State, Sequence[State]],
) -> Union[M, State]:
    """Merge a ``GraphDef`` with ``State``(s) into a ``Module``, or merge multiple ``State`` objects.

    Args:
        first: Either a ``GraphDef[M]`` to reconstruct a ``Module``, or a ``State`` dictionary.
        *rest: Remaining ``State`` dictionaries (or sequences of ``State`` dictionaries) to merge.

    Returns:
        Reconstructed ``Module`` if ``first`` is a ``GraphDef``, otherwise a merged ``State``.
    """
    if isinstance(first, GraphDef):
        state_list: list[State] = []
        for item in rest:
            if isinstance(item, State):
                state_list.append(item)
            else:
                for sub in item:
                    state_list.append(sub)
        if not state_list:
            return first.merge(State())
        return first.merge(state_list[0], *state_list[1:])

    all_states: list[State] = [first]
    for item in rest:
        if isinstance(item, State):
            all_states.append(item)
        else:
            for sub in item:
                all_states.append(sub)
    return state_merge(*all_states)


def clone(module: M) -> M:
    """Return an independent deep clone of ``module`` via ``split`` and ``merge``.

    Args:
        module: Source ``Module`` instance to clone.

    Returns:
        Cloned ``Module`` instance with independent ``Variable`` containers.
    """
    graphdef, current_state = split(module)
    return merge(graphdef, current_state)


def state(module: Module) -> State:
    """Return the hierarchical ``State`` of ``module``.

    Args:
        module: Target ``Module`` to extract state from.

    Returns:
        Hierarchical ``State`` dictionary of ``module``.
    """
    return module.state()


def update(
    module: Module,
    new_state: Optional[Union[State, Sequence[State]]] = None,
    *more_states: State,
    **kw_states: StateEntry,
) -> None:
    """Update ``module`` in-place with ``new_state`` and any additional states.

    Args:
        module: Target ``Module`` to update in-place.
        new_state: Primary ``State`` or sequence of ``State`` objects.
        *more_states: Additional ``State`` objects to merge and apply.
        **kw_states: Additional keyword state entries.
    """
    module.update(new_state, *more_states, **kw_states)


def iter_graph(
    node: Union[Module, State],
    prefix: Tuple[str, ...] = (),
    visited: Optional[set[int]] = None,
) -> Iterator[Tuple[Tuple[str, ...], Union[Module, AnyVariable, State]]]:
    """Iterate in depth-first order over all nodes (modules and variables) in ``node``.

    Args:
        node: Root ``Module`` or ``State`` to traverse.
        prefix: Tuple of string path keys leading to ``node``.
        visited: Optional set of visited object IDs used to prevent infinite cycles.

    Yields:
        Pairs ``(path_tuple, node_or_variable)`` for every module and variable in the graph.
    """
    seen = set() if visited is None else visited
    node_id = id(node)
    if node_id in seen:
        return
    seen.add(node_id)

    yield prefix, node
    if isinstance(node, Module):
        node._ensure_registries()
        for var_name, var_obj in node._variables.items():
            yield prefix + (var_name,), var_obj
        for child_name, child_mod in node._children.items():
            yield from iter_graph(child_mod, prefix + (child_name,), seen)
    elif isinstance(node, State):
        for key, val in node.items():
            if isinstance(val, State):
                yield from iter_graph(val, prefix + (key,), seen)
            elif isinstance(val, Variable):
                yield prefix + (key,), val


__all__ = [
    "ChildT",
    "GraphDef",
    "M",
    "Module",
    "VarT",
    "clone",
    "iter_graph",
    "merge",
    "split",
    "state",
    "update",
]
