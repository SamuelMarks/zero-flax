"""Variable containers and metadata tracking for the zero-flax NNX API.

This module defines the generic `Variable` class and its standard Flax NNX
subclasses (`Param`, `BatchStat`, `Cache`, `Intermediate`, `Rng`, `RngState`,
`RngCount`, `RngKey`, and `LoRAParam`) without using `Any`, `object`, or
dynamic attribute reflection.
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar, Union

from typing_extensions import TypeAlias

from zero_flax.types import ArrayLike, JaxArray, MetadataValue

VariableValue: TypeAlias = Union[JaxArray, ArrayLike, int, float, bool, str, None]
V = TypeVar("V", bound=VariableValue, covariant=True)


class Variable(Generic[V]):
    """Base class representing a stateful variable container in the NNX framework.

    Attributes:
        value: The underlying array or scalar value stored in the variable.
        metadata: Dictionary of static metadata key-value pairs associated with the variable.
    """

    value: V
    metadata: dict[str, MetadataValue]

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
        self.value = value
        self.metadata = dict(metadata)

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

    def replace(
        self,
        value: Optional[V] = None,
        **metadata: MetadataValue,
    ) -> Variable[V]:
        """Return a new Variable of the same type with an updated value or metadata.

        Args:
            value: Optional new value to store; retains ``self.value`` if ``None``.
            **metadata: Additional or updated metadata entries.

        Returns:
            New ``Variable`` instance with the updated value and merged metadata.
        """
        merged_meta = dict(self.metadata)
        merged_meta.update(metadata)
        new_val: V = self.value if value is None else value
        cls_type = type(self)
        return cls_type(new_val, **merged_meta)

    def copy(self) -> Variable[V]:
        """Return a shallow copy of this Variable container.

        Returns:
            Copied ``Variable`` instance with the same value and metadata.
        """
        cls_type = type(self)
        return cls_type(self.value, **self.metadata)

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
    """Variable subclass tracking the integer fold-in counter of an RNG stream.

    Attributes:
        value: Integer call counter value.
        metadata: Counter metadata dictionary.
    """

    def __init__(self, value: int = 0, **metadata: MetadataValue) -> None:
        """Initialize an RngCount counter variable.

        Args:
            value: Initial integer counter value. Defaults to ``0``.
            **metadata: Optional metadata entries.
        """
        super().__init__(int(value), **metadata)


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


__all__ = [
    "BatchStat",
    "Cache",
    "Intermediate",
    "LoRAParam",
    "Param",
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "V",
    "Variable",
    "VariableValue",
]
