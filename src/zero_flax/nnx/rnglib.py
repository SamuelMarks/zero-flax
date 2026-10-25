"""Random number generator stream management (`Rngs` and `RngStream`) for zero-flax NNX.

This module implements deterministic PRNG key streams backed by `zero_jax.random`
(`PRNGKey`, `fold_in`, `split`) without `Any`, `object`, or `__getattr__` reflection.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Optional, Union

from zero_jax import numpy as jnp
from zero_jax import random as jrandom

from zero_flax.nnx.module import Module
from zero_flax.nnx.variables import Rng, RngCount, RngKey, RngState
from zero_flax.types import JaxArray, PRNGKeyLike, ensure_array


def _collection_tag_hash(collection: str) -> int:
    """Compute a deterministic 31-bit integer tag for an RNG collection name.

    Args:
        collection: Name of the RNG stream (e.g., ``"params"``, ``"dropout"``).

    Returns:
        Deterministic non-negative integer hash in ``[0, 2**31 - 1]``.
    """
    return (
        0 if collection == "default" else int(zlib.adler32(collection.encode("utf-8")) & 0x7FFFFFFF)
    )


def _to_prng_key(seed_or_key: PRNGKeyLike) -> JaxArray:
    """Convert an integer seed or key-like value into a 2-element ``uint32`` PRNG key array.

    Args:
        seed_or_key: Integer seed, pair of ints, existing ``JaxArray`` key, or ``None``.

    Returns:
        2-element ``JaxArray`` PRNG key.
    """
    if seed_or_key is None:
        res_zero: JaxArray = jrandom.PRNGKey(0)
        return res_zero
    if isinstance(seed_or_key, int):
        res_int: JaxArray = jrandom.PRNGKey(int(seed_or_key))
        return res_int
    arr = ensure_array(seed_or_key)
    if arr.ndim == 0:
        res_scalar: JaxArray = jrandom.PRNGKey(0)
        return res_scalar
    return arr


class RngStream(Module):
    """Stateful PRNG key stream tracking a base `RngKey` and a monotonically increasing `RngCount`.

    Attributes:
        key: Registered ``RngKey`` variable holding the base 2-element PRNG key.
        count: Registered ``RngCount`` variable tracking the number of keys generated.
        tag: Stream collection name (e.g., ``"default"``, ``"params"``, ``"dropout"``).
    """

    key: RngKey
    count: RngCount
    tag: str

    def __init__(
        self,
        key: PRNGKeyLike = 0,
        *,
        tag: str = "default",
        count: int = 0,
    ) -> None:
        """Initialize an RngStream with a seed or key and optional stream tag.

        Args:
            key: Initial integer seed or PRNG key array. Defaults to ``0``.
            tag: Stream identifier string. Defaults to ``"default"``.
            count: Initial call count. Defaults to ``0``.
        """
        super().__init__(tag=tag)
        self.tag = tag
        prng_key = _to_prng_key(key)
        self.key = self.register_variable("key", RngKey(prng_key, tag=tag))
        self.count = self.register_variable("count", RngCount(count, tag=tag))
        self.finish_init()

    def __call__(self) -> JaxArray:
        """Generate the next deterministic PRNG key by folding ``count`` into ``key`` and incrementing ``count``.

        Returns:
            Newly folded 2-element ``JaxArray`` PRNG key.
        """
        current_count = int(self.count.value)
        self.count.value = current_count + 1
        folded: JaxArray = jrandom.fold_in(self.key.value, current_count)
        return folded

    def fork(self, split: Optional[Union[int, Sequence[int]]] = None) -> JaxArray:
        """Fork a new PRNG key (or batch of ``split`` keys) from this stream.

        Args:
            split: Optional number or shape of subkeys to split the forked key into.

        Returns:
            Forked ``JaxArray`` key of shape ``(2,)`` or ``(*split, 2)``.
        """
        next_key = self()
        if split is not None:
            if isinstance(split, int):
                split_keys: JaxArray = jrandom.split(next_key, int(split))
                return split_keys
            shape_tuple = tuple(int(d) for d in split)
            total = 1
            for d in shape_tuple:
                total *= d
            flat_keys = jrandom.split(next_key, total)
            reshaped: JaxArray = jnp.reshape(flat_keys, (*shape_tuple, 2))
            return reshaped
        return next_key


class Rngs(Module):
    """Collection of named `RngStream` instances for parameter initialization, dropout, and carry generation.

    Attributes:
        _streams: Dictionary mapping stream names (``"default"``, ``"params"``, ``"dropout"``, etc.)
            to child ``RngStream`` modules.
    """

    _streams: dict[str, RngStream]

    def __init__(
        self,
        default: PRNGKeyLike = None,
        **rngs: PRNGKeyLike,
    ) -> None:
        """Initialize an Rngs collection with an optional default seed/key and named stream seeds/keys.

        Args:
            default: Optional seed or PRNG key for the ``"default"`` fallback stream.
            **rngs: Named seeds or PRNG keys (e.g., ``params=0``, ``dropout=1``).
        """
        super().__init__()
        self._streams = {}
        if default is not None or not rngs:
            seed_val: PRNGKeyLike = 0 if default is None else default
            self._register_stream("default", seed_val)
        for stream_name, stream_seed in rngs.items():
            self._register_stream(stream_name, stream_seed)
        self.finish_init()

    def _register_stream(
        self, name: str, seed_or_stream: Union[PRNGKeyLike, RngStream]
    ) -> RngStream:
        """Create or register an ``RngStream`` under ``name``.

        Args:
            name: Stream collection name.
            seed_or_stream: Seed/key value or existing ``RngStream`` instance.

        Returns:
            Registered ``RngStream`` child module.
        """
        if isinstance(seed_or_stream, RngStream):
            stream = seed_or_stream
        else:
            stream = RngStream(seed_or_stream, tag=name)
        self._streams[name] = stream
        self.register_child(name, stream)
        return stream

    def _get_or_create_stream(self, name: str) -> RngStream:
        """Return the ``RngStream`` for ``name``, falling back to ``"default"`` if registered.

        Args:
            name: Requested stream name.

        Returns:
            Matching or fallback ``RngStream``.

        Raises:
            KeyError: If neither ``name`` nor ``"default"`` exists in ``self._streams``.
        """
        if name in self._streams:
            return self._streams[name]
        if "default" in self._streams:
            return self._streams["default"]
        raise KeyError(f"No RNG stream named {name!r} and no 'default' stream configured in Rngs.")

    def __getitem__(self, name: str) -> RngStream:
        """Retrieve an ``RngStream`` by ``name`` (falling back to ``"default"``).

        Args:
            name: Stream name (e.g., ``"params"``, ``"dropout"``).

        Returns:
            Corresponding ``RngStream`` instance.
        """
        return self._get_or_create_stream(name)

    def __call__(self, name: str = "default") -> JaxArray:
        """Draw the next PRNG key from stream ``name`` (or ``"default"``).

        When falling back to the ``"default"`` stream for a specific non-default
        ``name``, a deterministic collection tag hash is folded into the drawn key
        so different collections (such as ``"params"`` and ``"dropout"``) never collide.

        Args:
            name: Stream name to draw from. Defaults to ``"default"``.

        Returns:
            Next 2-element ``JaxArray`` PRNG key.
        """
        stream = self._get_or_create_stream(name)
        raw_key = stream()
        if name not in self._streams and name != "default":
            tag_hash = _collection_tag_hash(name)
            folded_tag: JaxArray = jrandom.fold_in(raw_key, tag_hash)
            return folded_tag
        return raw_key

    @property
    def default(self) -> RngStream:
        """Return the ``"default"`` ``RngStream``.

        Returns:
            The ``"default"`` ``RngStream`` instance.
        """
        return self._get_or_create_stream("default")

    @property
    def params(self) -> RngStream:
        """Return the ``"params"`` ``RngStream`` (or ``"default"`` fallback).

        Returns:
            The ``"params"`` ``RngStream`` instance.
        """
        return self._get_or_create_stream("params")

    @property
    def dropout(self) -> RngStream:
        """Return the ``"dropout"`` ``RngStream`` (or ``"default"`` fallback).

        Returns:
            The ``"dropout"`` ``RngStream`` instance.
        """
        return self._get_or_create_stream("dropout")

    @property
    def carry(self) -> RngStream:
        """Return the ``"carry"`` ``RngStream`` (or ``"default"`` fallback).

        Returns:
            The ``"carry"`` ``RngStream`` instance.
        """
        return self._get_or_create_stream("carry")

    def fork(self, split: Optional[Union[int, Sequence[int]]] = None) -> JaxArray:
        """Fork a key (or ``split`` keys) from the default stream.

        Args:
            split: Optional number or shape of subkeys to split into.

        Returns:
            Forked ``JaxArray`` PRNG key(s).
        """
        first_stream = (
            self._streams["default"]
            if "default" in self._streams
            else next(iter(self._streams.values()))
        )
        return first_stream.fork(split=split)


def resolve_rng_key(
    rngs: Union[Rngs, RngStream, PRNGKeyLike],
    collection: str = "params",
    fold_index: int = 0,
) -> JaxArray:
    """Resolve an ``Rngs``, ``RngStream``, or raw seed/key into a concrete ``JaxArray`` PRNG key.

    Args:
        rngs: ``Rngs`` container, ``RngStream`` instance, raw seed/key, or ``None``.
        collection: Stream name to query when ``rngs`` is an ``Rngs`` instance.
        fold_index: Deterministic fold-in index used when ``rngs`` is a static key or ``None``.

    Returns:
        2-element ``JaxArray`` PRNG key.
    """
    if isinstance(rngs, Rngs):
        return rngs(collection)
    if isinstance(rngs, RngStream):
        return rngs()
    base = _to_prng_key(rngs)
    tag_hash = _collection_tag_hash(collection)
    if tag_hash != 0:
        base = jrandom.fold_in(base, tag_hash)
    if fold_index > 0:
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
