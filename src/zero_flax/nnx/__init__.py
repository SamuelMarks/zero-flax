from __future__ import annotations
import ml_switcheroo
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
import numpy
from zero_jax import numpy as jnp

Array = Any
ArrayLike = Any


class AxisName:
    pass


class PrecisionLike:
    pass


class PaddingLike:
    pass


class Dtype:
    pass


class Shape:
    pass


class Axes:
    pass


class Size:
    pass


class Axis:
    pass


class DotGeneralT:
    pass


class MaxFun:
    pass


class filterlib:
    class Filter:
        pass


class rnglib:
    class Rngs:
        pass


class variables:
    class Variable:
        pass


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
from .state import Variable, Param, BatchStat, Rng, State, merge
from .module import Module
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
