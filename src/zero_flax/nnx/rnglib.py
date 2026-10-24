"""Random number generator stream management (`Rngs` and `RngStream`) for zero-flax NNX.

This module implements deterministic PRNG key streams backed by `zero_jax.random`
(`PRNGKey`, `fold_in`, `split`) without `Any`, `object`, or reflection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

from zero_jax import numpy as jnp
from zero_jax import random as jrandom

from zero_flax.nnx.module import Module
from zero_flax.nnx.variables import Rng, RngCount, RngKey, RngState
from zero_flax.types import JaxArray, PRNGKeyLike, ensure_array


def _normalize_key(seed_or_key: PRNGKeyLike) -> JaxArray:
    """Convert an integer seed or key array into a 2-element ``JaxArray`` PRNG key.

    Args:
        seed_or_key: Integer seed, tuple of ints, or existing PRNG key tensor.

    Returns:
        Normalized PRNG key ``JaxArray``.
    """
    if seed_or_key is None:
        res_zero: JaxArray = jrandom.PRNGKey(0)
        return res_zero
    if isinstance(seed_or_key, int):
        res_int: JaxArray = jrandom.PRNGKey(seed_or_key)
        return res_int
    return ensure_array(seed_or_key)


class RngStream(Module):
    """Stateful stream of PRNG keys supporting deterministic fold-in and fork splitting.

    Attributes:
        key: Base ``RngKey`` variable holding the root PRNG key tensor.
        count: ``RngCount`` variable tracking how many keys have been drawn.
        tag: String identifier name of this RNG stream (e.g., ``"params"``, ``"dropout"``).
    """

    key: RngKey
    count: RngCount
    tag: str

    def __init__(self, key: PRNGKeyLike = 0, tag: str = "default") -> None:
        """Initialize an RngStream with a seed or PRNG key and stream tag.

        Args:
            key: Integer seed or PRNG key tensor. Defaults to ``0``.
            tag: Stream collection name. Defaults to ``"default"``.
        """
        super().__init__()
        self.tag = tag
        self.key = self.register_variable("key", RngKey(_normalize_key(key), tag=tag))
        self.count = self.register_variable("count", RngCount(0, tag=tag))
        self.finish_init()

    def __call__(self) -> JaxArray:
        """Draw a fresh deterministic PRNG key by folding ``self.count`` into ``self.key``.

        Returns:
            Fresh PRNG key ``JaxArray``.
        """
        step = self.count.value
        self.count.value = step + 1
        derived: JaxArray = jrandom.fold_in(self.key.value, step)
        return derived

    def fork(self, split_size: Optional[Union[int, Sequence[int]]] = None) -> JaxArray:
        """Split a fresh PRNG key into multiple sub-keys for vectorized or parallel transforms.

        Args:
            split_size: Optional number of splits or shape of split keys.

        Returns:
            Split PRNG key ``JaxArray`` of shape ``(split_size, 2)`` or ``(*split_size, 2)``.
        """
        base_key = self()
        if split_size is None:
            return base_key
        if isinstance(split_size, int):
            split_res: JaxArray = jrandom.split(base_key, split_size)
            return split_res
        dims = tuple(int(d) for d in split_size)
        total = 1
        for dim in dims:
            total *= dim
        split_flat: JaxArray = jrandom.split(base_key, total)
        key_last_dim = split_flat.shape[-1] if split_flat.ndim >= 2 else 2
        reshaped: JaxArray = jnp.reshape(split_flat, (*dims, key_last_dim))
        return reshaped


class Rngs(Module):
    """Collection of named ``RngStream`` generators for NNX module initialization and execution.

    Attributes:
        _streams: Dictionary mapping stream names to ``RngStream`` instances.
    """

    _streams: dict[str, RngStream]

    def __init__(
        self,
        default: PRNGKeyLike = 0,
        **streams: PRNGKeyLike,
    ) -> None:
        """Initialize an Rngs collection with a default seed/key and optional named streams.

        Args:
            default: Default integer seed or PRNG key used for unnamed or fallback streams.
            **streams: Named PRNG seeds or keys (e.g., ``params=0``, ``dropout=1``).
        """
        super().__init__()
        self._streams = {}
        if default is not None:
            def_stream = RngStream(default, tag="default")
            self._streams["default"] = self.register_child("default", def_stream)
        for name, seed_val in streams.items():
            named_stream = RngStream(seed_val, tag=name)
            self._streams[name] = self.register_child(name, named_stream)
        self.finish_init()

    def _get_or_create_stream(self, name: str) -> RngStream:
        """Retrieve the named ``RngStream``, or fall back to ``"default"``.

        Args:
            name: Name of the requested RNG stream.

        Returns:
            Corresponding ``RngStream`` instance.

        Raises:
            KeyError: If neither ``name`` nor ``"default"`` is configured in ``self._streams``.
        """
        if name in self._streams:
            return self._streams[name]
        if "default" in self._streams:
            return self._streams["default"]
        raise KeyError(
            f"No RNG stream found for {name!r} and no 'default' stream configured on Rngs."
        )

    def __call__(self, name: str = "default") -> JaxArray:
        """Draw a fresh PRNG key from the specified stream (or ``"default"`` fallback).

        Args:
            name: Stream name to sample from. Defaults to ``"default"``.

        Returns:
            Fresh PRNG key ``JaxArray``.
        """
        return self._get_or_create_stream(name)()

    def default(self) -> JaxArray:
        """Draw a fresh PRNG key from the ``"default"`` stream.

        Returns:
            Fresh PRNG key ``JaxArray``.
        """
        return self("default")

    def params(self) -> JaxArray:
        """Draw a fresh PRNG key from the ``"params"`` stream.

        Returns:
            Fresh PRNG key ``JaxArray``.
        """
        return self("params")

    def dropout(self) -> JaxArray:
        """Draw a fresh PRNG key from the ``"dropout"`` stream.

        Returns:
            Fresh PRNG key ``JaxArray``.
        """
        return self("dropout")

    def __getitem__(self, name: str) -> RngStream:
        """Access the ``RngStream`` for ``name``.

        Args:
            name: Stream name.

        Returns:
            The ``RngStream`` associated with ``name``.
        """
        return self._get_or_create_stream(name)

    def has_stream(self, name: str) -> bool:
        """Check whether an explicit or fallback stream is available for ``name``.

        Args:
            name: Stream name to check.

        Returns:
            ``True`` if ``name`` or ``"default"`` exists in ``self._streams``.
        """
        return name in self._streams or "default" in self._streams


def resolve_rng_key(
    rngs: Optional[Union[Rngs, RngStream, bool]],
    collection: str = "params",
    fold_index: Optional[int] = None,
) -> JaxArray:
    """Resolve an ``Rngs``, ``RngStream``, or ``None`` into a fresh PRNG key.

    Args:
        rngs: Optional ``Rngs`` container, ``RngStream``, or boolean flag.
        collection: Target stream name (e.g., ``"params"`` or ``"dropout"``).
        fold_index: Optional deterministic fold-in index used when ``rngs`` is ``None``
            so multiple parameters initialized without an explicit ``Rngs`` receive
            distinct deterministic keys.

    Returns:
        Fresh PRNG key ``JaxArray``.
    """
    if isinstance(rngs, Rngs):
        return rngs(collection)
    if isinstance(rngs, RngStream):
        return rngs()
    base: JaxArray = jrandom.PRNGKey(0)
    if fold_index is not None:
        folded: JaxArray = jrandom.fold_in(base, fold_index)
        return folded
    return base


__all__ = [
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "RngStream",
    "Rngs",
    "resolve_rng_key",
]
