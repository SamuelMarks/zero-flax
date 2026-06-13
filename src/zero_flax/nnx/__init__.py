"""Flax NNX API compatibility layer and exports for zero-flax.

This module consolidates and exports the core components of the zero-flax NNX
implementation, including base module classes, state management, standard layers
(e.g., Linear, Conv, Embed), normalization techniques, and typing utilities.
"""

from __future__ import annotations


import typing as tp
from typing import (
    Any,
    Callable,
    Sequence,
    Union,
    Optional,
    Tuple,
    Iterable,
    Mapping,
    Literal,
    Type,
)
import collections.abc

Array = Any
ArrayLike = Any


import zero_chex as chex
import zero_optax as optax
import zero_jax as jax

M = tp.TypeVar("M")
A = tp.TypeVar("A")
UNSPECIFIED = None
_UNSPECIFIED = None
default_kernel_init = None
default_bias_init = None
default_embed_init = None
lax = Any
FrozenDict = Any
KeyArray = Any
RealNumeric = Any
LoRAParam = Any
dot_product_attention = None
from .state import Variable, Param, BatchStat, Rng, State
from .module import Module, split, merge, GraphDef
from .missing import (
    rnglib,
    filterlib,
    variables,
    AxisName,
    PrecisionLike,
    PaddingLike,
    Dtype,
    Shape,
    Axes,
    Size,
    Axis,
    DotGeneralT,
    MaxFun,
)

Rngs = rnglib.Rngs
from zero_jax.nn import initializers
from zero_jax.nn.initializers import zeros, ones, glorot_uniform, he_normal
from .containers import Dict as Dict, List as List, Sequential as Sequential
from .linear import Einsum as Einsum, Linear as Linear, LinearGeneral as LinearGeneral
from .normalization import (
    BatchNorm as BatchNorm,
    LayerNorm as LayerNorm,
    RMSNorm as RMSNorm,
)
from .stochastic import Dropout as Dropout
from .missing import (
    ConvTranspose as ConvTranspose,
    Jit as Jit,
    LoRA as LoRA,
    LoRALinear as LoRALinear,
    Pmap as Pmap,
    Remat as Remat,
    Scan as Scan,
    Vmap as Vmap,
    MultiHeadAttention as MultiHeadAttention,
)
from .layers import Dense as Dense, Conv as Conv, Embed as Embed
