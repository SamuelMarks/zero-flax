"""Core neural network layer re-exports (`Dense`, `Conv`, `ConvTranspose`, `Embed`, `MultiHeadDotProductAttention`).

This module consolidates and re-exports the fully implemented spatial convolution,
embedding, dense linear, and multi-head attention layers from their dedicated
`zero_flax.nnx` submodules.
"""

from __future__ import annotations

from zero_flax.nnx.attention import (
    MultiHeadAttention,
    MultiHeadDotProductAttention,
    dot_product_attention,
)
from zero_flax.nnx.conv import Conv, ConvTranspose, Embed
from zero_flax.nnx.linear import Dense, Linear

__all__ = [
    "Conv",
    "ConvTranspose",
    "Dense",
    "Embed",
    "Linear",
    "MultiHeadAttention",
    "MultiHeadDotProductAttention",
    "dot_product_attention",
]
