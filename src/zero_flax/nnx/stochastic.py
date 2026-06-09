from typing import Any, Sequence
from zero_flax.nnx.module import Module
import numpy as np


class Dropout(Module):
    """Create a dropout layer."""

    def __init__(
        self,
        rate: float,
        broadcast_dims: Sequence[int] = (),
        deterministic: bool = False,
        rng_collection: str = "dropout",
        rngs: Any = None,
    ):
        super().__init__()
        self.rate = rate
        self.broadcast_dims = broadcast_dims
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        self.rngs = rngs
        super().__setattr__("_is_initializing", False)

    def __call__(
        self, inputs: Any, deterministic: bool = None, rngs: Any = None
    ) -> Any:
        deterministic = self.deterministic if deterministic is None else deterministic
        if deterministic or self.rate == 0.0:
            return inputs

        # simple mock
        keep_prob = 1.0 - self.rate
        mask = np.random.binomial(1, keep_prob, size=np.shape(inputs))
        return inputs * mask / keep_prob
