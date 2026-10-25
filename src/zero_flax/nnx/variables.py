"""Variable containers and metadata tracking for the zero-flax NNX API.

This module defines the generic `Variable` class and its standard Flax NNX
subclasses (`Param`, `BatchStat`, `Cache`, `Intermediate`, `Rng`, `RngState`,
`RngCount`, `RngKey`, and `LoRAParam`) without using `Any`, `object`, or
dynamic attribute reflection.
"""

from __future__ import annotations

import copy
from typing import Generic, TypeVar, Union

from typing_extensions import TypeAlias
from zero_jax import numpy as jnp

from zero_flax.types import ArrayLike, JaxArray, MetadataValue

VariableValue: TypeAlias = Union[JaxArray, ArrayLike, int, float, bool, str, None]
V = TypeVar("V", bound=VariableValue)


class _UnsetSentinel:
    """Sentinel class distinguishing an omitted argument from an explicit ``None`` value."""


_UNSET = _UnsetSentinel()


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
    """

    value: V

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
        self.value = value

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
        return cls_type(new_val, **merged_meta)

    def copy(self) -> Variable[V]:
        """Return an independent copy of this Variable container with deep-copied metadata.

        Returns:
            Copied ``Variable`` instance with the same value and deep-copied metadata.
        """
        cls_type = type(self)
        return cls_type(self.value, **copy.deepcopy(self.metadata))

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
    "V",
    "Variable",
    "VariableBase",
    "VariableValue",
]
