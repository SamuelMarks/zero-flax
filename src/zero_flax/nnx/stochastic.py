"""Stochastic layers and operations for the NNX API.

This module provides stochastic components such as Dropout, which are
commonly used in neural networks for regularization.
"""

from zero_flax.nnx.module import Module
from typing import Any


class Dropout(Module):
    """A dropout layer.

    Applies dropout to the input data, which randomly zeroes some of the
    elements of the input tensor with probability `rate`.
    """

    def __init__(self, rate: float, *args: Any, **kwargs: Any) -> None:
        """Initializes the Dropout module.

        Args:
            rate: The probability of an element to be zeroed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.rate = rate
        self._is_initializing = False

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Applies dropout to the input.

        Args:
            x: The input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The input tensor `x` (currently acts as an identity function).
        """
        return x
