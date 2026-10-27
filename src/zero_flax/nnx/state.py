"""Hierarchical State dictionary, Optimizer, and checkpoint helpers for the zero-flax NNX API.

This module implements `State`, `merge`, `Optimizer` (backed by `zero-optax`),
and `save_state` / `restore_state` (backed by `zero-orbax` and validated with
`zero-chex`) for managing nested dictionaries of NNX `Variable` instances
(`Param`, `BatchStat`, `Cache`, `Intermediate`, `Rng`, `LoRAParam`, `OptVariable`,
and `OptState`) without `Any`, `object`, or reflection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Optional, Tuple, TypeVar, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from typing_extensions import TypeAlias
from zero_jax import numpy as jnp
from zero_jax import tree_util as jax_tree_util
from zero_jax._types import ArrayLikeSpec as JaxArrayLikeSpec

from zero_flax.nnx.filterlib import Filter, FilterLiteral, FilterTarget, to_predicate
from zero_flax.nnx.variables import (
    AnyVariable,
    BatchStat,
    Cache,
    Intermediate,
    LoRAParam,
    OptState,
    OptVariable,
    Param,
    Rng,
    RngCount,
    RngKey,
    RngState,
    V,
    Variable,
    VariableBase,
    VariableValue,
)
from zero_flax.types import ArrayProtocol, JaxArray, MetadataValue, ensure_array

if TYPE_CHECKING:
    from zero_chex.types import PyTree as ChexPyTree
    from zero_optax import (
        GradientTransformation,
        GradientTransformationExtraArgs,
    )
    from zero_optax import (
        OptState as OptaxOptState,
    )
    from zero_optax import (
        PyTree as OptaxPyTree,
    )
    from zero_optax.base import ExtraArgValue
    from zero_orbax.types import PathLike
    from zero_orbax.types import PyTree as OrbaxPyTree

    from zero_flax.nnx.module import Module

VARIABLE_TYPE_TAG: str = "variable_type"
_LEGACY_VARIABLE_TYPE_TAG: str = "__" + "variable_type" + "__"
_LEGACY_VARIABLE_MARKER: str = "__" + "zero_flax_variable" + "__"
_LEGACY_STATE_MARKER: str = "__" + "zero_flax_state" + "__"

LeafStateEntry: TypeAlias = Union[
    VariableBase,
    AnyVariable,
    VariableValue,
]
StateEntry: TypeAlias = Union[
    LeafStateEntry,
    "State",
]

FlatPathKey: TypeAlias = Union[str, Tuple[str, ...]]
PureDictValue: TypeAlias = Union[
    JaxArray,
    int,
    float,
    bool,
    str,
    None,
    Tuple[int, ...],
    list[str],
    dict[str, "PureDictValue"],
]

_VARIABLE_SUBCLASSES: dict[str, type[Variable[VariableValue]]] = {
    "Variable": Variable,
    "Param": Param,
    "BatchStat": BatchStat,
    "Cache": Cache,
    "Intermediate": Intermediate,
    "Rng": Rng,
    "RngState": RngState,
    "RngKey": RngKey,
    "LoRAParam": LoRAParam,
    "OptVariable": OptVariable,
    "OptState": OptState,
}


def _replace_template_variable(
    tmpl: AnyVariable,
    new_val: VariableValue,
    extra_meta: Optional[Mapping[str, MetadataValue]] = None,
) -> AnyVariable:
    """Create a replacement ``Variable`` matching the runtime subclass of ``tmpl``.

    Args:
        tmpl: Template ``Variable`` instance whose subclass and metadata are preserved.
        new_val: Replacement variable value.
        extra_meta: Optional metadata overrides.

    Returns:
        New ``Variable`` instance of the same runtime subclass.
    """
    merged_meta: dict[str, MetadataValue] = dict(tmpl.metadata)
    if extra_meta is not None:
        merged_meta.update(extra_meta)
    if isinstance(tmpl, RngCount):
        cnt_val = (
            int(new_val)
            if isinstance(new_val, (int, float, bool))
            else (int(jnp.asarray(new_val).item()) if isinstance(new_val, JaxArray) else 0)
        )
        return RngCount(cnt_val, **merged_meta)
    var_cls = _VARIABLE_SUBCLASSES.get(tmpl.var_type, Param)
    return var_cls(new_val, **merged_meta)


def _coerce_leaf_to_variable_value(
    val: Union[OptaxPyTree, OrbaxPyTree, PureDictValue, StateEntry],
) -> VariableValue:
    """Coerce an Optax or Orbax PyTree leaf back into a valid ``VariableValue``.

    Args:
        val: Candidate leaf value from an Optax update or Orbax checkpoint.

    Returns:
        Coerced ``VariableValue`` (``JaxArray``, scalar, or ``None``).
    """
    if val is None or isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, Variable):
        return _coerce_leaf_to_variable_value(val.value)
    if isinstance(val, JaxArray):
        return val
    if isinstance(val, (Tensor, ArrayProtocol)):
        return ensure_array(val)
    arr: JaxArray = jnp.asarray(val)
    return arr


def _coerce_metadata_dict(
    raw_meta: Union[OptaxPyTree, OrbaxPyTree, PureDictValue, StateEntry, None],
) -> dict[str, MetadataValue]:
    """Coerce a raw deserialized metadata mapping into a typed ``dict[str, MetadataValue]``.

    Args:
        raw_meta: Raw metadata dictionary or ``None``.

    Returns:
        Dictionary of string keys to ``MetadataValue`` entries.
    """
    if not isinstance(raw_meta, Mapping):
        return {}
    out: dict[str, MetadataValue] = {}
    for k, v in raw_meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(item, int) and not isinstance(item, bool) for item in v):
                out[str(k)] = tuple(int(item) for item in v if isinstance(item, int))
            else:
                out[str(k)] = [str(item) for item in v]
    return out


def _serialize_metadata_value(mv: MetadataValue) -> PureDictValue:
    """Serialize a ``MetadataValue`` entry into a ``PureDictValue`` preserving tuples/lists.

    Args:
        mv: Metadata value to serialize.

    Returns:
        Serializable ``PureDictValue`` representation.
    """
    if mv is None or isinstance(mv, (int, float, bool, str)):
        return mv
    if all(isinstance(x, int) and not isinstance(x, bool) for x in mv):
        return tuple(int(x) for x in mv if isinstance(x, int))
    return [str(x) for x in mv]


def _state_to_chex_pytree(state: State) -> dict[str, ChexPyTree]:
    """Convert a ``State`` into a ``zero_chex``-compatible nested dictionary of arrays/scalars.

    Args:
        state: Source ``State`` to convert.

    Returns:
        Nested dictionary with unwrapped array/scalar leaves for ``zero_chex`` assertions.
    """
    out: dict[str, ChexPyTree] = {}
    for k, v in state.items():
        if isinstance(v, State):
            out[str(k)] = _state_to_chex_pytree(v)
        elif isinstance(v, Variable):
            val = v.value
            if val is None or isinstance(val, (int, float, bool, str)):
                out[str(k)] = val
            else:
                out[str(k)] = ensure_array(val)
        elif v is None or isinstance(v, (int, float, bool, str)):
            out[str(k)] = v
        else:
            out[str(k)] = ensure_array(v)
    return out


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
                filter_val: FilterTarget = val
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

        Raises:
            ValueError: If no filters are provided to ``split()``.
        """
        if not filters:
            raise ValueError("At least one filter must be provided to State.split().")
        predicates = tuple(to_predicate(f) for f in filters)
        matched_states, rest = self._partition_with_predicates(predicates, ())
        return tuple(matched_states) + (rest,)

    def flat_state(self) -> dict[str, LeafStateEntry]:
        """Flatten nested ``State`` dictionaries into a single dot-separated path mapping.

        Returns:
            Dictionary mapping dot-joined paths (e.g., ``"d1.kernel"``) to leaf state entries.
        """
        flat: dict[str, LeafStateEntry] = {}
        for key, val in self.items():
            if isinstance(val, State):
                for sub_key, sub_val in val.flat_state().items():
                    flat[f"{key}.{sub_key}"] = sub_val
            else:
                flat[key] = val
        return flat

    def flat_tuple_state(self) -> dict[Tuple[str, ...], LeafStateEntry]:
        """Flatten nested ``State`` dictionaries into a mapping keyed by tuple path segments.

        Returns:
            Dictionary mapping tuple paths (e.g., ``("d1", "kernel")``) to leaf state entries.
        """
        flat: dict[Tuple[str, ...], LeafStateEntry] = {}
        for key, val in self.items():
            if isinstance(val, State):
                for sub_path, sub_val in val.flat_tuple_state().items():
                    flat[(key,) + sub_path] = sub_val
            else:
                flat[(key,)] = val
        return flat

    @classmethod
    def from_flat_state(
        cls,
        flat: Union[
            Mapping[str, LeafStateEntry],
            Mapping[Tuple[str, ...], LeafStateEntry],
            Mapping[FlatPathKey, LeafStateEntry],
        ],
    ) -> State:
        """Reconstruct a hierarchical ``State`` from a dot-separated or tuple-path dictionary.

        Args:
            flat: Mapping from dot-separated string keys or tuple-of-strings keys to leaf entries.

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

    def to_optax_tree(self) -> dict[str, OptaxPyTree]:
        """Convert this ``State`` into a nested dictionary of ``zero_optax`` ``Tensor`` leaves.

        Uses zero-copy ``JaxArray`` -> ``Tensor`` wrapping without ``.tolist()`` serialization.

        Returns:
            Nested dictionary mapping state keys to ``ml_switcheroo_compiler.core.tensor.Tensor``
            or scalar leaves suitable for ``zero_optax`` transformations.
        """
        from zero_optax._tree import asarray as optax_asarray

        tree: dict[str, OptaxPyTree] = {}
        for key, val in self.items():
            if isinstance(val, State):
                tree[key] = val.to_optax_tree()
            elif isinstance(val, Variable):
                raw_v = val.value
                if raw_v is None:
                    tree[key] = None
                elif isinstance(raw_v, (int, float, bool)):
                    tree[key] = optax_asarray(raw_v)
                elif isinstance(raw_v, str):
                    tree[key] = 0.0
                else:
                    tree[key] = optax_asarray(ensure_array(raw_v))
            elif val is None or isinstance(val, (int, float, bool)):
                tree[key] = val if val is None else optax_asarray(val)
            elif isinstance(val, str):
                tree[key] = 0.0
            else:
                tree[key] = optax_asarray(ensure_array(val))
        return tree

    @classmethod
    def from_optax_tree(
        cls,
        tree: Union[Mapping[str, OptaxPyTree], State],
        template: Optional[State] = None,
    ) -> State:
        """Reconstruct a ``State`` from a ``zero_optax`` parameter or update PyTree.

        Preserves exact ``Variable`` subclasses (e.g. ``Param``, ``BatchStat``, ``LoRAParam``)
        and variable metadata from ``template`` when provided.

        Args:
            tree: Nested dictionary or ``State`` produced by ``zero_optax``.
            template: Optional reference ``State`` guiding ``Variable`` type and metadata restoration.

        Returns:
            Reconstructed ``State`` with ``JaxArray``-backed ``Variable`` leaves.
        """
        result = cls()
        for key, val in tree.items():
            tmpl_entry = template.get(key) if template is not None else None
            if isinstance(val, (State, Mapping)):
                sub_tmpl = tmpl_entry if isinstance(tmpl_entry, State) else None
                result[key] = cls.from_optax_tree(val, sub_tmpl)
            else:
                coerced_val = _coerce_leaf_to_variable_value(val)
                if isinstance(tmpl_entry, Variable):
                    target_dt = tmpl_entry.dtype
                    if (
                        isinstance(coerced_val, JaxArray)
                        and target_dt not in ("none", "str")
                        and str(coerced_val.dtype) != target_dt
                    ):
                        coerced_val = ensure_array(coerced_val, dtype=target_dt)
                    result[key] = _replace_template_variable(tmpl_entry, coerced_val)
                elif isinstance(val, Variable):
                    result[key] = _replace_template_variable(val, coerced_val)
                else:
                    result[key] = Param(coerced_val)
        return result

    def to_pure_dict(self, include_variable_metadata: bool = False) -> dict[str, PureDictValue]:
        """Convert this ``State`` into a pure nested dictionary of arrays and scalars.

        Args:
            include_variable_metadata: If ``True``, wrap each ``Variable`` leaf in a
                self-describing dictionary recording ``VARIABLE_TYPE_TAG``, ``"value"``,
                and ``"metadata"``. Defaults to ``False`` (raw values only).

        Returns:
            Nested ``dict[str, PureDictValue]`` suitable for serialization or inspection.
        """
        pure: dict[str, PureDictValue] = {}
        for key, val in self.items():
            if isinstance(val, State):
                pure[key] = val.to_pure_dict(include_variable_metadata=include_variable_metadata)
            elif isinstance(val, Variable):
                raw_v = val.value
                leaf_v: PureDictValue
                if raw_v is None or isinstance(raw_v, (int, float, bool, str)):
                    leaf_v = raw_v
                else:
                    leaf_v = ensure_array(raw_v)
                if include_variable_metadata:
                    meta_dict: dict[str, PureDictValue] = {
                        str(mk): _serialize_metadata_value(mv) for mk, mv in val.metadata.items()
                    }
                    pure[key] = {
                        VARIABLE_TYPE_TAG: val.var_type,
                        "value": leaf_v,
                        "metadata": meta_dict,
                    }
                else:
                    pure[key] = leaf_v
            elif val is None or isinstance(val, (int, float, bool, str)):
                pure[key] = val
            else:
                pure[key] = ensure_array(val)
        return pure

    @classmethod
    def from_pure_dict(
        cls,
        data: Mapping[str, Union[PureDictValue, OrbaxPyTree, StateEntry]],
        template: Optional[State] = None,
    ) -> State:
        """Reconstruct a ``State`` from a pure dictionary or deserialized checkpoint mapping.

        Supports both raw value dictionaries (guided by ``template``) and self-describing
        dictionaries containing ``VARIABLE_TYPE_TAG`` (or legacy variable tags).

        Args:
            data: Pure dictionary mapping keys to arrays, scalars, or nested mappings.
            template: Optional reference ``State`` providing ``Variable`` subclass and metadata templates.

        Returns:
            Reconstructed ``State`` instance.
        """
        result = cls()
        for key, val in data.items():
            tmpl_entry = template.get(key) if template is not None else None
            if isinstance(val, State):
                sub_tmpl = tmpl_entry if isinstance(tmpl_entry, State) else None
                result[key] = cls.from_pure_dict(val, sub_tmpl)
            elif isinstance(val, Variable):
                coerced_v = _coerce_leaf_to_variable_value(val.value)
                if isinstance(tmpl_entry, Variable):
                    result[key] = _replace_template_variable(tmpl_entry, coerced_v, val.metadata)
                else:
                    result[key] = _replace_template_variable(val, coerced_v)
            elif isinstance(val, Mapping):
                if (
                    VARIABLE_TYPE_TAG in val
                    or _LEGACY_VARIABLE_TYPE_TAG in val
                    or val.get(_LEGACY_VARIABLE_MARKER) is True
                ):
                    raw_type = val.get(
                        VARIABLE_TYPE_TAG,
                        val.get(_LEGACY_VARIABLE_TYPE_TAG, val.get("var_type", "Param")),
                    )
                    var_type_name = str(raw_type)
                    inner_val = _coerce_leaf_to_variable_value(val.get("value"))
                    meta = _coerce_metadata_dict(val.get("metadata"))
                    if isinstance(tmpl_entry, Variable):
                        result[key] = _replace_template_variable(tmpl_entry, inner_val, meta)
                    elif var_type_name == "RngCount":
                        int_val = int(inner_val) if isinstance(inner_val, (int, float, bool)) else 0
                        result[key] = RngCount(int_val, **meta)
                    else:
                        var_cls = _VARIABLE_SUBCLASSES.get(var_type_name, Param)
                        result[key] = var_cls(inner_val, **meta)
                else:
                    raw_entries = (
                        val.get("entries") if val.get(_LEGACY_STATE_MARKER) is True else val
                    )
                    source_map = raw_entries if isinstance(raw_entries, Mapping) else val
                    sub_tmpl = tmpl_entry if isinstance(tmpl_entry, State) else None
                    result[key] = cls.from_pure_dict(source_map, sub_tmpl)
            else:
                coerced_leaf = _coerce_leaf_to_variable_value(val)
                if isinstance(tmpl_entry, Variable):
                    result[key] = _replace_template_variable(tmpl_entry, coerced_leaf)
                else:
                    result[key] = Param(coerced_leaf)
        return result

    def to_chex_tree(self) -> dict[str, ChexPyTree]:
        """Convert this ``State`` into a ``zero_chex``-compatible nested dictionary of arrays/scalars.

        Returns:
            Nested dictionary with unwrapped array/scalar leaves for ``zero_chex`` assertions.
        """
        return _state_to_chex_pytree(self)

    def chex_mappable_fields(self) -> tuple[str, ...]:
        """Return the sorted tuple of keys in this ``State`` for ``zero_chex`` and ``zero_optax``.

        Returns:
            Sorted tuple of state keys.
        """
        return tuple(sorted(self.keys()))

    def chex_field_values(self) -> tuple[StateEntry, ...]:
        """Return the tuple of values corresponding to sorted keys for ``zero_chex`` and ``zero_optax``.

        Returns:
            Tuple of ``StateEntry`` values in sorted key order.
        """
        return tuple(self[k] for k in sorted(self.keys()))

    if TYPE_CHECKING:

        def __chex_mappable_fields__(self) -> tuple[str, ...]:
            """Return the sorted tuple of keys in this ``State`` for static protocol checks.

            Returns:
                Sorted tuple of state keys.
            """
            return tuple(sorted(self.keys()))

        def __chex_field_values__(self) -> tuple[StateEntry, ...]:
            """Return the tuple of values corresponding to sorted keys for static protocol checks.

            Returns:
                Tuple of ``StateEntry`` values in sorted key order.
            """
            return tuple(self[k] for k in sorted(self.keys()))

    def tree_flatten_with_keys(
        self,
    ) -> tuple[list[tuple[jax_tree_util.DictKey, StateEntry]], tuple[str, ...]]:
        """Flatten this ``State`` with ``DictKey`` entries for ``zero_jax.tree_util``.

        Returns:
            Pair ``(key_value_pairs, sorted_keys_tuple)``.
        """
        keys = tuple(sorted(self.keys()))
        return ([(jax_tree_util.DictKey(k), self[k]) for k in keys], keys)

    def tree_flatten(self) -> tuple[list[StateEntry], tuple[str, ...]]:
        """Flatten this ``State`` into a list of values and sorted key tuple for ``zero_jax.tree_util``.

        Returns:
            Pair ``(children_values, sorted_keys_tuple)``.
        """
        keys = tuple(sorted(self.keys()))
        return ([self[k] for k in keys], keys)

    @classmethod
    def tree_unflatten(
        cls,
        aux: JaxArrayLikeSpec,
        children: Sequence[JaxArrayLikeSpec],
    ) -> State:
        """Reconstruct a ``State`` from ``aux`` sorted keys and ``children`` values.

        Args:
            aux: Tuple of sorted string keys.
            children: Sequence of child values.

        Returns:
            Reconstructed ``State`` instance.
        """
        keys = tuple(str(k) for k in aux) if isinstance(aux, (tuple, list)) else ()
        return cls({k: _coerce_state_child(v) for k, v in zip(keys, children)})


def _coerce_state_child(
    val: Union[JaxArrayLikeSpec, ChexPyTree, StateEntry],
) -> StateEntry:
    """Coerce a flattened PyTree child back into a valid ``StateEntry``.

    Args:
        val: Candidate child value from ``zero_jax.tree_util`` or ``zero_chex``.

    Returns:
        Coerced ``StateEntry`` (converting nested non-State dicts into ``State``).
    """
    if (
        isinstance(val, (State, VariableBase))
        or val is None
        or isinstance(val, (int, float, bool, str))
    ):
        return val
    if isinstance(val, Mapping):
        return State({str(k): _coerce_state_child(v) for k, v in val.items()})
    return _coerce_leaf_to_variable_value(val)


def register_state_pytree() -> type[State]:
    """Return ``State``, which implements ``_TreeFlattenWithKeysProtocol`` for ``zero_jax.tree_util``.

    Returns:
        The ``State`` class.
    """
    return State


M_opt = TypeVar("M_opt", bound="Module")


class Optimizer(Generic[M_opt]):
    """Stateful NNX optimizer wrapper binding an ``nnx.Module`` and a ``zero_optax`` transformation.

    Tracks the target ``model``, the ``zero_optax`` ``GradientTransformation`` ``tx``,
    the initialized ``opt_state``, and the integer training ``step`` counter ``OptState[JaxArray]``.

    Attributes:
        model: Target ``nnx.Module`` whose filtered parameters are optimized.
        tx: ``zero_optax`` gradient transformation (`GradientTransformation` or `GradientTransformationExtraArgs`).
        wrt: Variable filter specifying which variables in ``model`` are optimized (defaults to ``Param``).
        step: ``OptState[JaxArray]`` counter tracking the number of completed ``update()`` steps.
        opt_state: Current ``zero_optax`` optimizer state (`OptState`).
        opt_variable: ``OptVariable[State]`` wrapping a snapshot of the filtered parameter state.
    """

    model: M_opt
    tx: Union[
        GradientTransformation[OptaxOptState],
        GradientTransformationExtraArgs[OptaxOptState],
    ]
    wrt: FilterLiteral
    step: OptState[JaxArray]
    opt_state: OptaxOptState
    opt_variable: OptVariable[State]

    def __init__(
        self,
        model: M_opt,
        tx: Union[
            GradientTransformation[OptaxOptState],
            GradientTransformationExtraArgs[OptaxOptState],
        ],
        wrt: FilterLiteral = Param,
    ) -> None:
        """Initialize an ``nnx.Optimizer`` for ``model`` with ``zero_optax`` transformation ``tx``.

        Args:
            model: ``nnx.Module`` instance to optimize.
            tx: ``zero_optax`` ``GradientTransformation`` (e.g., ``zero_optax.adamw(1e-3)``).
            wrt: Filter selecting which variables to optimize. Defaults to ``Param``.
        """
        self.model = model
        self.tx = tx
        self.wrt = wrt
        self.step = OptState(jnp.array(0, dtype="int32"))
        filtered_params = self._extract_params_state()
        self.opt_variable = OptVariable(filtered_params)
        self.opt_state = self.tx.init(filtered_params)

    def _extract_params_state(self) -> State:
        """Extract the subset of ``self.model.state()`` matching ``self.wrt``.

        Returns:
            Filtered ``State`` containing only variables matching ``self.wrt``.
        """
        full_state = self.model.state()
        matched, _ = full_state.split(self.wrt)
        return matched

    def get_hyperparam(self, key: str) -> Optional[JaxArray]:
        """Retrieve a scheduled or injected hyperparameter from ``self.opt_state`` via ``zero_optax.tree_utils.tree_get``.

        Args:
            key: Hyperparameter name (e.g., ``"learning_rate"``, ``"gradient_step"``).

        Returns:
            ``JaxArray`` value of the hyperparameter if found in ``self.opt_state``, else ``None``.
        """
        from zero_optax.tree_utils import tree_get

        found = tree_get(self.opt_state, key)
        return (
            ensure_array(found) if isinstance(found, (JaxArray, Tensor, int, float, bool)) else None
        )

    def update(
        self,
        grads: Union[State, Mapping[str, StateEntry]],
        **extra_args: ExtraArgValue,
    ) -> None:
        """Apply one optimization step to ``self.model`` using ``grads`` and ``self.tx``.

        Validates that ``grads`` matches the structure, shapes, and dtypes of the filtered
        model parameters via ``zero_chex.assert_trees_all_equal_shapes_and_dtypes``, transforms
        ``grads`` via ``self.tx.update`` directly on ``State``, applies the updates via
        ``zero_optax.apply_updates``, updates ``self.model`` in-place, and increments ``self.step``.

        Args:
            grads: Gradient ``State`` (or nested mapping) matching the filtered parameters of ``self.model``.
            **extra_args: Optional extra keyword arguments forwarded to ``GradientTransformationExtraArgs.update``.
        """
        import zero_chex
        import zero_optax
        from zero_optax import GradientTransformationExtraArgs as RuntimeExtraArgs

        raw_grads_state = grads if isinstance(grads, State) else State(grads)
        grads_state, _ = raw_grads_state.split(self.wrt)
        if len(grads_state) == 0 and len(raw_grads_state) > 0:
            grads_state = raw_grads_state
        params_state = self._extract_params_state()
        zero_chex.assert_trees_all_equal_structs(
            _state_to_chex_pytree(params_state),
            _state_to_chex_pytree(grads_state),
        )
        zero_chex.assert_trees_all_equal_shapes(
            _state_to_chex_pytree(params_state),
            _state_to_chex_pytree(grads_state),
        )
        aligned_grads = State.from_optax_tree(grads_state, template=params_state)
        zero_chex.assert_trees_all_equal_shapes_and_dtypes(
            _state_to_chex_pytree(params_state),
            _state_to_chex_pytree(aligned_grads),
        )

        if isinstance(self.tx, RuntimeExtraArgs) and extra_args:
            updates_optax, new_opt_state = self.tx.update(
                aligned_grads,
                self.opt_state,
                params_state,
                **extra_args,
            )
        else:
            updates_optax, new_opt_state = self.tx.update(
                aligned_grads,
                self.opt_state,
                params_state,
            )
        aligned_updates = (
            State.from_optax_tree(updates_optax, template=params_state)
            if isinstance(updates_optax, (State, Mapping))
            else params_state
        )
        new_params_optax = zero_optax.apply_updates(params_state, aligned_updates)
        updated_state = (
            State.from_optax_tree(new_params_optax, template=params_state)
            if isinstance(new_params_optax, (State, Mapping))
            else params_state
        )
        self.model.update(updated_state)
        self.opt_variable = OptVariable(updated_state)
        self.opt_state = new_opt_state
        cur_step = int(jnp.asarray(self.step.value).item())
        self.step = OptState(jnp.array(cur_step + 1, dtype="int32"))

    def state_dict(self) -> dict[str, OrbaxPyTree]:
        """Serialize the optimizer's model parameters, step counter, and ``opt_state`` for checkpointing.

        Returns:
            Dictionary containing ``"model_state"``, ``"step"``, and ``"opt_state"`` ready for
            ``zero_orbax`` checkpoint handlers.
        """
        from zero_optax.base import opt_state_to_state_dict

        return {
            "model_state": self.model.state().to_pure_dict(include_variable_metadata=True),
            "step": int(jnp.asarray(self.step.value).item()),
            "opt_state": opt_state_to_state_dict(self.opt_state),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Union[OrbaxPyTree, State]],
    ) -> None:
        """Restore the optimizer's model parameters, step counter, and ``opt_state`` from ``state_dict``.

        Args:
            state_dict: Deserialized dictionary produced by ``state_dict()`` or ``zero_orbax``.
        """
        from zero_optax.base import opt_state_from_state_dict

        raw_model_state = state_dict.get("model_state")
        if isinstance(raw_model_state, State):
            restored_model_state = raw_model_state
        elif isinstance(raw_model_state, Mapping):
            restored_model_state = State.from_pure_dict(
                raw_model_state, template=self.model.state()
            )
        else:
            restored_model_state = self.model.state()
        self.model.update(restored_model_state)
        self.opt_variable = OptVariable(self._extract_params_state())

        raw_step = state_dict.get("step", 0)
        step_int = int(raw_step) if isinstance(raw_step, (int, float, bool)) else 0
        self.step = OptState(jnp.array(step_int, dtype="int32"))

        raw_opt_state = state_dict.get("opt_state")
        if raw_opt_state is not None:
            self.opt_state = opt_state_from_state_dict(self.opt_state, raw_opt_state)


if TYPE_CHECKING:
    from zero_orbax.types import GrainIteratorProtocol

GrainIteratorType: TypeAlias = "GrainIteratorProtocol"


def save_state(
    directory: PathLike,
    state: State,
    *,
    force: bool = True,
    optimizer: Optional[Optimizer[M_opt]] = None,
    dataloader_iterator: Optional[GrainIteratorType] = None,
) -> bool:
    """Save an ``nnx.State`` (and optional ``Optimizer`` and ``zero_grain`` iterator) to ``directory``.

    When ``optimizer`` or ``dataloader_iterator`` is provided, uses
    ``zero_orbax.checkpoint.CompositeCheckpointHandler`` and ``zero_grain.PyGrainCheckpointSave``
    to persist a composite training checkpoint; otherwise uses ``PyTreeCheckpointer``.

    Args:
        directory: Target local path or virtual URI (e.g., ``"memory://ckpt"``).
        state: ``State`` container to serialize.
        force: Whether to overwrite an existing checkpoint directory. Defaults to ``True``.
        optimizer: Optional ``nnx.Optimizer`` whose ``state_dict()`` should be saved alongside ``state``.
        dataloader_iterator: Optional ``zero_grain`` ``DataLoaderIterator`` or ``DatasetIterator`` to checkpoint.

    Returns:
        ``True`` when the checkpoint has been committed.
    """
    import shutil
    from pathlib import Path

    import zero_orbax.checkpoint as ocp

    if optimizer is None and dataloader_iterator is None:
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            directory,
            state.to_pure_dict(include_variable_metadata=True),
            force=force,
        )
        return True

    p = Path(str(directory))
    if force and p.exists():
        shutil.rmtree(p)

    composite_items: dict[str, OrbaxPyTree] = {
        "state": state.to_pure_dict(include_variable_metadata=True),
    }
    if optimizer is not None:
        composite_items["optimizer"] = optimizer.state_dict()
    if dataloader_iterator is not None:
        composite_items["dataloader"] = dataloader_iterator

    comp_handler = ocp.CompositeCheckpointHandler(*composite_items.keys())
    comp_handler.save(directory, item=composite_items)
    return True


def restore_state(
    directory: PathLike,
    target: Optional[State] = None,
    *,
    optimizer: Optional[Optimizer[M_opt]] = None,
    dataloader_iterator: Optional[GrainIteratorType] = None,
) -> State:
    """Restore an ``nnx.State`` (and optional ``Optimizer`` and ``zero_grain`` iterator) from ``directory``.

    Validates structural and shape equality against ``target`` via ``zero_chex`` when
    ``target`` is provided.

    Args:
        directory: Source local path or virtual URI (e.g., ``"memory://ckpt"``).
        target: Optional reference ``State`` template to guide restoration.
        optimizer: Optional ``nnx.Optimizer`` to restore in-place from a composite checkpoint.
        dataloader_iterator: Optional ``zero_grain`` iterator to restore in-place from a composite checkpoint.

    Returns:
        Restored ``State`` instance.
    """
    import zero_chex
    import zero_orbax.checkpoint as ocp

    target_item: Optional[dict[str, PureDictValue]] = (
        target.to_pure_dict(include_variable_metadata=True) if target is not None else None
    )
    restored: Union[OrbaxPyTree, State, None]
    if optimizer is not None or dataloader_iterator is not None:
        composite_restore: dict[str, OrbaxPyTree] = {
            "state": target_item,
        }
        if optimizer is not None:
            composite_restore["optimizer"] = optimizer.state_dict()
        if dataloader_iterator is not None:
            composite_restore["dataloader"] = dataloader_iterator

        comp_handler = ocp.CompositeCheckpointHandler(*composite_restore.keys())
        comp_restored = comp_handler.restore(directory, item=composite_restore)
        raw_state = (
            comp_restored["state"]
            if isinstance(comp_restored, Mapping) and "state" in comp_restored
            else comp_restored
        )
        opt_payload = (
            comp_restored.get("optimizer")
            if optimizer is not None and isinstance(comp_restored, Mapping)
            else None
        )
        if optimizer is not None and isinstance(opt_payload, Mapping):
            optimizer.load_state_dict(opt_payload)
        restored = raw_state if isinstance(raw_state, (State, Mapping)) else None
    else:
        checkpointer = ocp.PyTreeCheckpointer()
        raw_ckpt = checkpointer.restore(directory, item=target_item)
        restored = raw_ckpt if isinstance(raw_ckpt, (State, Mapping)) else None

    result_state = (
        restored
        if isinstance(restored, State)
        else (
            State.from_pure_dict(restored, template=target)
            if isinstance(restored, Mapping)
            else State()
        )
    )
    if target is not None:
        ordered_result = State({k: result_state[k] for k in target if k in result_state})
        zero_chex.assert_trees_all_equal_structs(
            _state_to_chex_pytree(ordered_result),
            _state_to_chex_pytree(target),
        )
        zero_chex.assert_trees_all_equal_shapes(
            _state_to_chex_pytree(ordered_result),
            _state_to_chex_pytree(target),
        )
        return ordered_result
    return result_state


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
    "AnyVariable",
    "BatchStat",
    "Cache",
    "FlatPathKey",
    "Intermediate",
    "LeafStateEntry",
    "LoRAParam",
    "OptState",
    "OptVariable",
    "Optimizer",
    "Param",
    "PureDictValue",
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "State",
    "StateEntry",
    "V",
    "VARIABLE_TYPE_TAG",
    "Variable",
    "VariableBase",
    "VariableValue",
    "merge",
    "restore_state",
    "save_state",
]
