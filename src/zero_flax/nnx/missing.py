"""Compatibility re-export module for `zero_flax.nnx.missing`.

All components previously stubbed in `missing.py` (`ConvTranspose`, `LoRA`,
`LoRALinear`, `MultiHeadAttention`, `Jit`, `Remat`, `Vmap`, `Scan`, `Pmap`,
`filterlib`, `rnglib`, `variables`, `DotGeneralT`, `Initializer`, `PaddingLike`,
`PrecisionLike`, `Shape`, `Dtype`, `Axis`, `Axes`, `Size`, `AxisName`, `MaxFun`)
are now fully implemented in their respective modules and re-exported here.
"""

from __future__ import annotations

from zero_flax.nnx import filterlib as filterlib
from zero_flax.nnx import rnglib as rnglib
from zero_flax.nnx import variables as variables
from zero_flax.nnx.attention import (
    MultiHeadAttention,
    dot_product_attention,
    ones_init,
)
from zero_flax.nnx.conv import ConvTranspose
from zero_flax.nnx.linear import (
    LoRA,
    LoRALinear,
    default_bias_init,
    default_kernel_init,
)
from zero_flax.nnx.transforms import Jit, Pmap, Remat, Scan, Vmap
from zero_flax.nnx.variables import LoRAParam
from zero_flax.types import (
    Array,
    Axes,
    Axis,
    AxisName,
    DotGeneralT,
    Dtype,
    Initializer,
    MaxFun,
    PaddingLike,
    PrecisionLike,
    Shape,
    Size,
    lax,
)

default_a_initializer = default_kernel_init
default_b_initializer = default_bias_init

__all__ = [
    "Array",
    "Axes",
    "Axis",
    "AxisName",
    "ConvTranspose",
    "DotGeneralT",
    "Dtype",
    "Initializer",
    "Jit",
    "LoRA",
    "LoRALinear",
    "LoRAParam",
    "MaxFun",
    "MultiHeadAttention",
    "PaddingLike",
    "Pmap",
    "PrecisionLike",
    "Remat",
    "Scan",
    "Shape",
    "Size",
    "Vmap",
    "default_a_initializer",
    "default_b_initializer",
    "default_bias_init",
    "default_kernel_init",
    "dot_product_attention",
    "filterlib",
    "lax",
    "ones_init",
    "rnglib",
    "variables",
]
