"""Variable containers and metadata tracking for the zero-flax NNX API.

This module defines the generic `Variable` class and its standard Flax NNX
subclasses (`Param`, `BatchStat`, `Cache`, `Intermediate`, `Rng`, `RngState`,
`RngCount`, `RngKey`, and `LoRAParam`) without using `Any`, `object`, or
dynamic attribute reflection.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union

from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_ir import PartitionSpec
from typing_extensions import TypeAlias
from zero_jax import numpy as jnp
from zero_jax import tree_util as jax_tree_util
from zero_jax._types import ArrayLikeSpec as JaxArrayLikeSpec
from zero_jax.sharding import NamedSharding, Sharding, SingleDeviceSharding

from zero_flax.types import ArrayLike, JaxArray, MetadataValue

if TYPE_CHECKING:
    from zero_chex.types import PyTree as ChexPyTree

VariableValue: TypeAlias = Union[JaxArray, ArrayLike, int, float, bool, str, None]
ShardingSpec: TypeAlias = Union[NamedSharding, PartitionSpec, Sharding, MetadataValue]
V = TypeVar("V", bound=VariableValue)


class _UnsetSentinel:
    """Sentinel class distinguishing an omitted argument from an explicit ``None`` value."""


_UNSET = _UnsetSentinel()


def _resolve_partition_spec(sharding_meta: MetadataValue) -> Optional[PartitionSpec]:
    """Convert a metadata sharding value into a ``PartitionSpec`` if applicable.

    Args:
        sharding_meta: Metadata sharding entry (tuple/list of axis names or single axis name).

    Returns:
        Constructed ``PartitionSpec`` or ``None`` if not convertible.
    """
    if isinstance(sharding_meta, str):
        return PartitionSpec((sharding_meta,))
    if isinstance(sharding_meta, (tuple, list)):
        axes: list[Optional[str]] = [
            item if isinstance(item, str) else None
            for item in sharding_meta
            if isinstance(item, str) or item is None
        ]
        return PartitionSpec(axes)
    return None


def _attach_sharding_to_array(arr: JaxArray, sharding_spec: ShardingSpec) -> JaxArray:
    """Attach ``sharding_spec`` to ``arr`` in-place and return ``arr``.

    Args:
        arr: Target ``JaxArray`` instance.
        sharding_spec: Sharding descriptor or metadata tuple/string.

    Returns:
        ``JaxArray`` with updated ``_sharding`` attribute.
    """
    if isinstance(sharding_spec, (NamedSharding, PartitionSpec, Sharding)):
        arr._sharding = sharding_spec
        return arr
    pspec = _resolve_partition_spec(sharding_spec)
    if pspec is not None:
        arr._sharding = pspec
    return arr


def _values_equal(lhs: VariableValue, rhs: VariableValue) -> bool:
    """Compare two ``VariableValue`` instances for structural and numerical equality.

    Args:
        lhs: Left-hand variable value.
        rhs: Right-hand variable value.

    Returns:
        ``True`` if ``lhs`` and ``rhs`` are equal in type, shape, and elements.
    """
    if lhs is None or rhs is None:
        return lhs is None and rhs is None
    if isinstance(lhs, (int, float, bool, str)) and isinstance(rhs, (int, float, bool, str)):
        if isinstance(lhs, bool) != isinstance(rhs, bool):
            return False
        return lhs == rhs
    lhs_arr = jnp.asarray(lhs)
    rhs_arr = jnp.asarray(rhs)
    if lhs_arr.shape != rhs_arr.shape:
        return False
    return bool(jnp.all(lhs_arr == rhs_arr))


class VariableBase:
    """Invariant-free base class for all NNX Variable containers.

    Attributes:
        metadata: Dictionary of static metadata key-value pairs associated with the variable.
    """

    metadata: dict[str, MetadataValue]

    def __init__(self, **metadata: MetadataValue) -> None:
        """Initialize base Variable metadata.

        Args:
            **metadata: Arbitrary metadata key-value pairs attached to this variable.
        """
        self.metadata = copy.deepcopy(dict(metadata))

    def chex_mappable_fields(self) -> tuple[str, ...]:
        """Return the ordered tuple of mappable field names for ``zero_chex`` and ``zero_optax``.

        Returns:
            Single-element tuple ``("value",)``.
        """
        return ("value",)

    def chex_field_values(self) -> tuple[VariableValue, ...]:
        """Return the ordered tuple of mappable field values for ``zero_chex`` and ``zero_optax``.

        Returns:
            Single-element tuple containing ``None`` on base instances or ``(self.value,)`` on subclasses.
        """
        return (None,)

    if TYPE_CHECKING:

        def __chex_mappable_fields__(self) -> tuple[str, ...]:
            """Return the ordered tuple of mappable field names for static protocol checks.

            Returns:
                Single-element tuple ``("value",)``.
            """
            return ("value",)

        def __chex_field_values__(self) -> tuple[VariableValue, ...]:
            """Return the ordered tuple of mappable field values for static protocol checks.

            Returns:
                Single-element tuple containing the variable value.
            """
            return (None,)

    def has_metadata(self, key: str) -> bool:
        """Check whether ``key`` exists in ``self.metadata``.

        Args:
            key: Metadata key name to look up.

        Returns:
            ``True`` if ``key`` is present in ``self.metadata``, otherwise ``False``.
        """
        return key in self.metadata

    def get_metadata(self, key: str, default: MetadataValue = None) -> MetadataValue:
        """Retrieve a metadata value by ``key``.

        Args:
            key: Metadata key name to look up.
            default: Fallback value if ``key`` is not present.

        Returns:
            Metadata value associated with ``key`` or ``default``.
        """
        return self.metadata.get(key, default)


class Variable(VariableBase, Generic[V]):
    """Base class representing a stateful variable container in the NNX framework.

    Attributes:
        value: The underlying array or scalar value stored in the variable.
        metadata: Dictionary of static metadata key-value pairs associated with the variable.
        _active_sharding: Optional ``NamedSharding``, ``PartitionSpec``, or ``Sharding`` object.
    """

    value: V
    _active_sharding: Optional[Union[NamedSharding, PartitionSpec, Sharding]]

    def __init__(
        self,
        value: V,
        **metadata: MetadataValue,
    ) -> None:
        """Initialize a Variable container with a value and optional metadata.

        Args:
            value: The initial array or scalar value to store.
            **metadata: Arbitrary metadata key-value pairs attached to this variable.
        """
        super().__init__(**metadata)
        self._active_sharding = None
        self.value = value
        sh_meta = self.metadata.get("sharding")
        if isinstance(self.value, JaxArray):
            if sh_meta is not None and (
                self.value._sharding is None
                or isinstance(self.value._sharding, SingleDeviceSharding)
            ):
                _attach_sharding_to_array(self.value, sh_meta)
            if isinstance(
                self.value._sharding, (NamedSharding, PartitionSpec, Sharding)
            ) and not isinstance(self.value._sharding, SingleDeviceSharding):
                self._active_sharding = self.value._sharding
        if sh_meta is not None and self._active_sharding is None:
            resolved_pspec = _resolve_partition_spec(sh_meta)
            if resolved_pspec is not None:
                self._active_sharding = resolved_pspec

    def chex_field_values(self) -> tuple[VariableValue, ...]:
        """Return the single-element tuple ``(self.value,)`` for PyTree traversal.

        Returns:
            Tuple ``(self.value,)``.
        """
        return (self.value,)

    def tree_flatten_with_keys(
        self,
    ) -> tuple[list[tuple[jax_tree_util.GetAttrKey, V]], dict[str, MetadataValue]]:
        """Flatten this ``Variable`` with ``GetAttrKey('value')`` and metadata dictionary for ``zero_jax.tree_util``.

        Returns:
            Pair ``([(GetAttrKey('value'), self.value)], copy.deepcopy(self.metadata))``.
        """
        return (
            [(jax_tree_util.GetAttrKey("value"), self.value)],
            copy.deepcopy(self.metadata),
        )

    def tree_flatten(self) -> tuple[list[V], dict[str, MetadataValue]]:
        """Flatten this ``Variable`` into ``([self.value], metadata)`` for ``zero_jax.tree_util``.

        Returns:
            Pair ``([self.value], copy.deepcopy(self.metadata))``.
        """
        return ([self.value], copy.deepcopy(self.metadata))

    @classmethod
    def tree_unflatten(
        cls,
        aux: JaxArrayLikeSpec,
        children: Sequence[JaxArrayLikeSpec],
    ) -> VariableBase:
        """Reconstruct a ``Variable`` subclass instance from ``aux`` metadata and ``children``.

        Args:
            aux: Auxiliary metadata dictionary.
            children: Single-element sequence containing the leaf value.

        Returns:
            Reconstructed ``VariableBase`` instance of type ``cls``.
        """
        meta = _coerce_aux_metadata(aux)
        leaf_val = _coerce_variable_leaf_value(children[0]) if len(children) > 0 else None
        return _instantiate_variable_subclass(cls, leaf_val, meta)

    @property
    def sharding(self) -> Optional[Union[NamedSharding, PartitionSpec, Sharding, MetadataValue]]:
        """Return the sharding specification associated with this Variable.

        Returns:
            Active ``NamedSharding``, ``PartitionSpec``, ``Sharding``, or metadata ``"sharding"`` value.
        """
        if self._active_sharding is not None:
            return self._active_sharding
        if "sharding" in self.metadata:
            return self.metadata["sharding"]
        if isinstance(self.value, JaxArray):
            return self.value.sharding
        return None

    def set_sharding(
        self,
        sharding_spec: Union[NamedSharding, PartitionSpec, Sharding, MetadataValue],
    ) -> Variable[V]:
        """Set or update the sharding metadata and array sharding on this Variable in-place.

        Args:
            sharding_spec: ``NamedSharding``, ``PartitionSpec``, ``Sharding``, or axis name tuple.

        Returns:
            ``self`` with updated sharding metadata and array sharding.
        """
        if isinstance(sharding_spec, NamedSharding):
            self._active_sharding = sharding_spec
            self.metadata["sharding"] = [str(ax) for ax in sharding_spec.spec if ax is not None]
        elif isinstance(sharding_spec, PartitionSpec):
            self._active_sharding = sharding_spec
            self.metadata["sharding"] = [str(ax) for ax in sharding_spec if ax is not None]
        elif isinstance(sharding_spec, Sharding):
            self._active_sharding = sharding_spec
        else:
            self.metadata["sharding"] = sharding_spec
            self._active_sharding = _resolve_partition_spec(sharding_spec)
        if isinstance(self.value, JaxArray):
            _attach_sharding_to_array(self.value, sharding_spec)
        return self

    def replace(
        self,
        value: Union[V, _UnsetSentinel] = _UNSET,
        **metadata: MetadataValue,
    ) -> Variable[V]:
        """Return a new Variable of the same type with an updated value or metadata.

        Args:
            value: Optional new value to store; retains ``self.value`` if omitted.
            **metadata: Additional or updated metadata entries.

        Returns:
            New ``Variable`` instance with the updated value and merged metadata.
        """
        merged_meta = copy.deepcopy(self.metadata)
        merged_meta.update(copy.deepcopy(metadata))
        new_val: V = self.value if isinstance(value, _UnsetSentinel) else value
        cls_type = type(self)
        replaced = cls_type(new_val, **merged_meta)
        if self._active_sharding is not None and "sharding" not in metadata:
            replaced.set_sharding(self._active_sharding)
        return replaced

    def copy(self) -> Variable[V]:
        """Return an independent copy of this Variable container with deep-copied metadata.

        Returns:
            Copied ``Variable`` instance with the same value and deep-copied metadata.
        """
        cls_type = type(self)
        copied = cls_type(self.value, **copy.deepcopy(self.metadata))
        if self._active_sharding is not None:
            copied.set_sharding(self._active_sharding)
        return copied

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape tuple of the underlying variable value.

        Returns:
            Tuple of integer dimension sizes (or ``()`` for scalar/``None`` values).
        """
        if self.value is None or isinstance(self.value, (int, float, bool, str)):
            return ()
        arr = jnp.asarray(self.value)
        return tuple(int(d) for d in arr.shape)

    @property
    def dtype(self) -> str:
        """Return the canonical dtype string of the underlying variable value.

        Returns:
            Canonical dtype string (e.g., ``"float32"``, ``"int32"``, ``"bool"``).
        """
        if self.value is None:
            return "none"
        if isinstance(self.value, bool):
            return "bool"
        if isinstance(self.value, int):
            return "int32"
        if isinstance(self.value, float):
            return "float32"
        if isinstance(self.value, str):
            return "str"
        arr = jnp.asarray(self.value)
        return str(arr.dtype).lower().removeprefix("dtype.")

    @property
    def var_type(self) -> str:
        """Return the runtime class name of this Variable container.

        Returns:
            Class name string (e.g., ``"Param"``, ``"BatchStat"``, ``"OptState"``).
        """
        return type(self).__name__

    def same_value_and_metadata(self, other: VariableBase) -> bool:
        """Check whether ``other`` has the exact same runtime class, value, and metadata.

        Args:
            other: Another ``VariableBase`` instance to compare against.

        Returns:
            ``True`` if ``type(self) is type(other)``, values match, and metadata matches.
        """
        if type(self) is not type(other) or not isinstance(other, Variable):
            return False
        if self.metadata != other.metadata:
            return False
        return _values_equal(self.value, other.value)

    def __repr__(self) -> str:
        """Return a string representation of the Variable.

        Returns:
            Formatted string ``"<ClassName>(<value>)"``.
        """
        return f"{type(self).__name__}({self.value})"


class Param(Variable[V]):
    """Variable subclass representing a trainable neural network parameter.

    Attributes:
        value: Trainable parameter array or scalar value.
        metadata: Parameter metadata dictionary.
    """


class BatchStat(Variable[V]):
    """Variable subclass representing a non-trainable batch normalization statistic.

    Attributes:
        value: Running statistic array or scalar value.
        metadata: Batch statistic metadata dictionary.
    """


class Cache(Variable[V]):
    """Variable subclass representing an autoregressive attention decoding cache entry.

    Attributes:
        value: Cached tensor or index value.
        metadata: Cache metadata dictionary.
    """


class Intermediate(Variable[V]):
    """Variable subclass representing an intermediate activation value recorded during execution.

    Attributes:
        value: Intermediate activation tensor or scalar value.
        metadata: Intermediate metadata dictionary.
    """


class Rng(Variable[V]):
    """Base Variable subclass representing random number generator state.

    Attributes:
        value: Underlying PRNG state or key value.
        metadata: RNG metadata dictionary.
    """


class RngState(Rng[V]):
    """Variable subclass representing a stateful PRNG value.

    Attributes:
        value: Underlying PRNG state value.
        metadata: RNG state metadata dictionary.
    """


class RngCount(Rng[int]):
    """Rng subclass storing the monotonic integer counter of an RNG stream.

    Attributes:
        value: Integer step count of the RNG stream.
        metadata: RNG counter metadata dictionary.
    """

    def __init__(self, value: Union[int, ArrayLike] = 0, **metadata: MetadataValue) -> None:
        """Initialize an RngCount counter variable.

        Args:
            value: Initial integer counter value (or scalar array). Defaults to ``0``.
            **metadata: Optional metadata entries.
        """
        int_val = (
            int(value) if isinstance(value, (int, float, bool)) else int(jnp.asarray(value).item())
        )
        super().__init__(int_val, **metadata)


class RngKey(Rng[JaxArray]):
    """Variable subclass storing a JAX/Switcheroo PRNG key tensor.

    Attributes:
        value: PRNG key array.
        metadata: Key metadata dictionary.
    """


class LoRAParam(Param[V]):
    """Param subclass representing a Low-Rank Adaptation (LoRA) trainable parameter.

    Attributes:
        value: LoRA projection matrix tensor.
        metadata: LoRA parameter metadata dictionary.
    """


class OptVariable(Variable[V]):
    """Variable subclass representing a wrapped optimizer state leaf in ``nnx.Optimizer``.

    Attributes:
        value: Optimizer state array or scalar value.
        metadata: Optimizer variable metadata dictionary.
    """


class OptState(OptVariable[V]):
    """Variable subclass representing optimizer step and moment state in ``nnx.Optimizer``.

    Attributes:
        value: Optimizer state tensor or step counter.
        metadata: Optimizer state metadata dictionary.
    """


AnyVariable: TypeAlias = Union[
    Variable[VariableValue],
    Variable[JaxArray],
    Variable[ArrayLike],
    Variable[int],
    Variable[float],
    Variable[bool],
    Variable[str],
    Variable[None],
]


def _coerce_variable_leaf_value(raw: Union[JaxArrayLikeSpec, ChexPyTree]) -> VariableValue:
    """Coerce a flattened PyTree child back into a valid ``VariableValue``.

    Args:
        raw: Child leaf from ``zero_jax.tree_util`` or ``zero_chex``.

    Returns:
        Coerced ``VariableValue`` (wrapping raw ``CompilerTensor`` in ``JaxArray``).
    """
    if isinstance(raw, CompilerTensor) and not isinstance(raw, JaxArray):
        return jnp.ndarray(raw)
    if isinstance(raw, (list, tuple)):
        arr_seq: JaxArray = jnp.asarray(raw)
        return arr_seq
    return raw if isinstance(raw, (int, float, bool, str, JaxArray)) else None


def _coerce_aux_metadata(aux: JaxArrayLikeSpec) -> dict[str, MetadataValue]:
    """Extract a ``dict[str, MetadataValue]`` from PyTree ``aux_data``.

    Args:
        aux: Auxiliary metadata returned by ``tree_flatten_with_keys``.

    Returns:
        Dictionary of variable metadata entries.
    """
    if not isinstance(aux, dict):
        return {}
    return {
        str(k): v
        for k, v in aux.items()
        if v is None or isinstance(v, (str, int, float, bool, tuple, list))
    }


_VARIABLE_CLASS_MAP: dict[str, type[Variable[VariableValue]]] = {
    "Variable": Variable,
    "Param": Param,
    "BatchStat": BatchStat,
    "Cache": Cache,
    "Intermediate": Intermediate,
    "Rng": Rng,
    "RngState": RngState,
    "LoRAParam": LoRAParam,
    "OptVariable": OptVariable,
    "OptState": OptState,
}


def _instantiate_variable_subclass(
    var_cls: type[VariableBase],
    leaf_val: VariableValue,
    meta: dict[str, MetadataValue],
) -> VariableBase:
    """Instantiate ``var_cls`` with ``leaf_val`` and ``meta``.

    Args:
        var_cls: Concrete ``VariableBase`` subclass to instantiate.
        leaf_val: Coerced variable value.
        meta: Variable metadata dictionary.

    Returns:
        Constructed ``VariableBase`` instance.
    """
    if var_cls is RngCount:
        int_val = int(leaf_val) if isinstance(leaf_val, (int, float, bool)) else 0
        return RngCount(int_val, **meta)
    if var_cls is RngKey:
        key_arr = leaf_val if isinstance(leaf_val, JaxArray) else jnp.asarray(0)
        return RngKey(key_arr, **meta)
    target_cls = _VARIABLE_CLASS_MAP.get(var_cls.__name__, Variable)
    return target_cls(leaf_val, **meta)


_ALL_VARIABLE_CLASSES: tuple[type[VariableBase], ...] = (
    Variable,
    Param,
    BatchStat,
    Cache,
    Intermediate,
    Rng,
    RngState,
    RngCount,
    RngKey,
    LoRAParam,
    OptVariable,
    OptState,
)


def register_variable_pytrees() -> tuple[type[VariableBase], ...]:
    """Return all standard ``VariableBase`` subclasses implementing ``_TreeFlattenWithKeysProtocol``.

    Returns:
        Tuple of registered ``VariableBase`` subclasses.
    """
    return _ALL_VARIABLE_CLASSES


__all__ = [
    "AnyVariable",
    "BatchStat",
    "Cache",
    "Intermediate",
    "LoRAParam",
    "OptState",
    "OptVariable",
    "Param",
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "ShardingSpec",
    "V",
    "Variable",
    "VariableBase",
    "VariableValue",
    "register_variable_pytrees",
]
