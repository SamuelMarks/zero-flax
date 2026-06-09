# ruff: noqa: E402, F811
from __future__ import annotations  # noqa
import typing as tp  # noqa
from typing import (  # noqa
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
import collections.abc  # noqa
import numpy  # noqa
from zero_jax import numpy as jnp  # noqa

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


class chex:
    class Array:
        pass

    class Numeric:
        pass

    class Scalar:
        pass


class core:
    class Shape:
        pass


class optax:
    class _src:
        class base:
            class GradientTransformationExtraArgs:
                pass


class base:
    class GradientTransformation:
        pass

    class Schedule:
        pass


class jax:
    class Array:
        pass

    class Device:
        pass

    class _src:
        class typing:
            class SupportsDType:
                pass


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

from .state import Variable, Param, BatchStat, Rng, State, merge  # noqa
from .module import Module  # noqa
from . import initializers  # noqa
from .initializers import zeros, ones, glorot_uniform, he_normal  # noqa


from .containers import Dict as Dict, List as List, Sequential as Sequential  # noqa: E402

from .linear import Einsum as Einsum, Linear as Linear, LinearGeneral as LinearGeneral  # noqa: F401, E402

from .normalization import (
    BatchNorm as BatchNorm,
    LayerNorm as LayerNorm,
    RMSNorm as RMSNorm,
)  # noqa: F401, E402, F811

from .stochastic import Dropout as Dropout  # noqa: F401, E402, F811
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
)  # noqa: F401, E402, F811

from .layers import (
    Dense as Dense,
    Conv as Conv,
    Embed as Embed,
)  # noqa: F401, E402
