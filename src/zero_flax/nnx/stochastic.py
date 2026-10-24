"""Stochastic regularization layers (`Dropout`) for the zero-flax NNX API.

This module implements `Dropout` with exact `flax_nnx_v0.12.10` parameter
signatures and Bernoulli keep-mask scaling during non-deterministic execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, Union

from zero_jax import numpy as jnp
from zero_jax import random as jrandom

from zero_flax.nnx.module import Module
from zero_flax.nnx.rnglib import Rngs, RngStream, resolve_rng_key
from zero_flax.types import ArrayLike, JaxArray, ensure_array


class Dropout(Module):
    """Stochastic Dropout regularization module zeroing elements with probability ``rate``.

    Attributes:
        rate: Dropout probability in ``[0.0, 1.0]``.
        broadcast_dims: Tuple of dimensions along which the dropout mask is broadcast.
        deterministic: Default flag controlling whether dropout is disabled.
        rng_collection: Name of the RNG stream used to sample masks (defaults to ``"dropout"``).
        rngs: Optional stored ``Rngs`` or ``RngStream`` instance.
    """

    rate: float
    broadcast_dims: Tuple[int, ...]
    deterministic: Optional[bool]
    rng_collection: str
    rngs: Optional[Union[Rngs, RngStream]]

    def __init__(
        self,
        rate: float,
        *,
        broadcast_dims: Sequence[int] = (),
        deterministic: Optional[bool] = False,
        rng_collection: str = "dropout",
        rngs: Optional[Union[Rngs, RngStream]] = None,
    ) -> None:
        """Initialize a Dropout module.

        Args:
            rate: Fraction of input units to drop (in ``[0.0, 1.0]``).
            broadcast_dims: Dimensions that will share the same dropout mask.
            deterministic: If ``True``, disables dropout and returns the input unchanged.
            rng_collection: RNG stream name used to draw Bernoulli masks.
            rngs: Optional ``Rngs`` or ``RngStream`` used when not passed to ``__call__``.

        Raises:
            ValueError: If ``rate`` is outside the interval ``[0.0, 1.0]``.
        """
        super().__init__()
        self.rate = float(rate)
        if self.rate < 0.0 or self.rate > 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0], got {self.rate}.")
        self.broadcast_dims = tuple(int(d) for d in broadcast_dims)
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        if isinstance(rngs, (Rngs, RngStream)):
            self.rngs = self.register_child("rngs", rngs)
        else:
            self.rngs = None
        self.finish_init()

    def __call__(
        self,
        x: ArrayLike,
        *,
        deterministic: Optional[bool] = None,
        rngs: Optional[Union[Rngs, RngStream]] = None,
    ) -> JaxArray:
        """Apply dropout to ``x`` when ``not deterministic`` and ``0 < rate <= 1``.

        Args:
            x: Input tensor to regularize.
            deterministic: Optional per-call override for ``self.deterministic``.
            rngs: Optional per-call RNG stream(s) override.

        Returns:
            Regularized ``JaxArray`` of the same shape as ``x``.
        """
        arr = ensure_array(x)
        det = self.deterministic if deterministic is None else deterministic
        if det or self.rate <= 0.0:
            return arr
        if self.rate >= 1.0:
            return jnp.zeros_like(arr)

        keep_prob = 1.0 - self.rate
        active_rngs = rngs if rngs is not None else self.rngs
        key = resolve_rng_key(active_rngs, self.rng_collection)
        mask_shape = list(arr.shape)
        for dim in self.broadcast_dims:
            norm_dim = dim if dim >= 0 else arr.ndim + dim
            mask_shape[norm_dim] = 1
        keep_mask = jrandom.bernoulli(key, p=keep_prob, shape=tuple(mask_shape))
        res: JaxArray = jnp.where(keep_mask, arr / keep_prob, 0.0)
        return res


__all__ = ["Dropout"]
