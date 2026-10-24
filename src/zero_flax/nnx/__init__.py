"""Flax NNX API compatibility layer and public exports for zero-flax.

This module consolidates and exports the complete zero-flax NNX surface,
including `Module`, `GraphDef`, `State`, `Variable` subclasses, all 29
`flax_nnx_v0.12.10` layers/cells/wrappers, containers, functional transforms,
`rnglib`, and `filterlib`.
"""

from __future__ import annotations

from zero_jax.nn import (
    celu as celu,
)
from zero_jax.nn import (
    elu as elu,
)
from zero_jax.nn import (
    gelu as gelu,
)
from zero_jax.nn import (
    glu as glu,
)
from zero_jax.nn import (
    hard_sigmoid as hard_sigmoid,
)
from zero_jax.nn import (
    hard_tanh as hard_tanh,
)
from zero_jax.nn import (
    initializers as initializers,
)
from zero_jax.nn import (
    leaky_relu as leaky_relu,
)
from zero_jax.nn import (
    log_sigmoid as log_sigmoid,
)
from zero_jax.nn import (
    log_softmax as log_softmax,
)
from zero_jax.nn import (
    logsumexp as logsumexp,
)
from zero_jax.nn import (
    one_hot as one_hot,
)
from zero_jax.nn import (
    relu as relu,
)
from zero_jax.nn import (
    relu6 as relu6,
)
from zero_jax.nn import (
    selu as selu,
)
from zero_jax.nn import (
    sigmoid as sigmoid,
)
from zero_jax.nn import (
    silu as silu,
)
from zero_jax.nn import (
    soft_sign as soft_sign,
)
from zero_jax.nn import (
    softmax as softmax,
)
from zero_jax.nn import (
    softplus as softplus,
)
from zero_jax.nn import (
    standardize as standardize,
)
from zero_jax.nn import (
    swish as swish,
)

from zero_flax.nnx import filterlib as filterlib
from zero_flax.nnx import rnglib as rnglib
from zero_flax.nnx import state as state
from zero_flax.nnx import variables as variables
from zero_flax.nnx.attention import (
    MultiHeadAttention as MultiHeadAttention,
)
from zero_flax.nnx.attention import (
    MultiHeadDotProductAttention as MultiHeadDotProductAttention,
)
from zero_flax.nnx.attention import (
    PReLU as PReLU,
)
from zero_flax.nnx.attention import (
    RoPE as RoPE,
)
from zero_flax.nnx.attention import (
    dot_product_attention as dot_product_attention,
)
from zero_flax.nnx.containers import (
    Dict as Dict,
)
from zero_flax.nnx.containers import (
    List as List,
)
from zero_flax.nnx.containers import (
    Sequential as Sequential,
)
from zero_flax.nnx.conv import (
    Conv as Conv,
)
from zero_flax.nnx.conv import (
    ConvTranspose as ConvTranspose,
)
from zero_flax.nnx.conv import (
    Embed as Embed,
)
from zero_flax.nnx.filterlib import (
    All as All,
)
from zero_flax.nnx.filterlib import (
    AnyFilter as AnyFilter,
)
from zero_flax.nnx.filterlib import (
    Everything as Everything,
)
from zero_flax.nnx.filterlib import (
    Filter as Filter,
)
from zero_flax.nnx.filterlib import (
    Not as Not,
)
from zero_flax.nnx.filterlib import (
    Nothing as Nothing,
)
from zero_flax.nnx.filterlib import (
    OfType as OfType,
)
from zero_flax.nnx.filterlib import (
    PathContains as PathContains,
)
from zero_flax.nnx.linear import (
    Dense as Dense,
)
from zero_flax.nnx.linear import (
    Einsum as Einsum,
)
from zero_flax.nnx.linear import (
    Linear as Linear,
)
from zero_flax.nnx.linear import (
    LinearGeneral as LinearGeneral,
)
from zero_flax.nnx.linear import (
    LoRA as LoRA,
)
from zero_flax.nnx.linear import (
    LoRALinear as LoRALinear,
)
from zero_flax.nnx.module import (
    GraphDef as GraphDef,
)
from zero_flax.nnx.module import (
    Module as Module,
)
from zero_flax.nnx.module import (
    iter_graph as iter_graph,
)
from zero_flax.nnx.module import (
    merge as merge,
)
from zero_flax.nnx.module import (
    split as split,
)
from zero_flax.nnx.module import (
    update as update,
)
from zero_flax.nnx.normalization import (
    BatchNorm as BatchNorm,
)
from zero_flax.nnx.normalization import (
    GroupNorm as GroupNorm,
)
from zero_flax.nnx.normalization import (
    InstanceNorm as InstanceNorm,
)
from zero_flax.nnx.normalization import (
    LayerNorm as LayerNorm,
)
from zero_flax.nnx.normalization import (
    RMSNorm as RMSNorm,
)
from zero_flax.nnx.normalization import (
    SpectralNorm as SpectralNorm,
)
from zero_flax.nnx.normalization import (
    WeightNorm as WeightNorm,
)
from zero_flax.nnx.recurrent import (
    RNN as RNN,
)
from zero_flax.nnx.recurrent import (
    Bidirectional as Bidirectional,
)
from zero_flax.nnx.recurrent import (
    GRUCell as GRUCell,
)
from zero_flax.nnx.recurrent import (
    LSTMCell as LSTMCell,
)
from zero_flax.nnx.recurrent import (
    OptimizedLSTMCell as OptimizedLSTMCell,
)
from zero_flax.nnx.recurrent import (
    RNNCellBase as RNNCellBase,
)
from zero_flax.nnx.recurrent import (
    SimpleCell as SimpleCell,
)
from zero_flax.nnx.rnglib import (
    Rngs as Rngs,
)
from zero_flax.nnx.rnglib import (
    RngStream as RngStream,
)
from zero_flax.nnx.state import (
    BatchStat as BatchStat,
)
from zero_flax.nnx.state import (
    Cache as Cache,
)
from zero_flax.nnx.state import (
    Intermediate as Intermediate,
)
from zero_flax.nnx.state import (
    LoRAParam as LoRAParam,
)
from zero_flax.nnx.state import (
    Param as Param,
)
from zero_flax.nnx.state import (
    Rng as Rng,
)
from zero_flax.nnx.state import (
    RngCount as RngCount,
)
from zero_flax.nnx.state import (
    RngKey as RngKey,
)
from zero_flax.nnx.state import (
    RngState as RngState,
)
from zero_flax.nnx.state import (
    State as State,
)
from zero_flax.nnx.state import (
    Variable as Variable,
)
from zero_flax.nnx.stochastic import Dropout as Dropout
from zero_flax.nnx.transforms import (
    Jit as Jit,
)
from zero_flax.nnx.transforms import (
    Pmap as Pmap,
)
from zero_flax.nnx.transforms import (
    Remat as Remat,
)
from zero_flax.nnx.transforms import (
    Scan as Scan,
)
from zero_flax.nnx.transforms import (
    Vmap as Vmap,
)
from zero_flax.types import (
    Array as Array,
)
from zero_flax.types import (
    ArrayLike as _ArrayLike,
)
from zero_flax.types import (
    Axes as Axes,
)
from zero_flax.types import (
    Axis as Axis,
)
from zero_flax.types import (
    AxisName as AxisName,
)
from zero_flax.types import (
    DotGeneralT as DotGeneralT,
)
from zero_flax.types import (
    DType as DType,
)
from zero_flax.types import (
    Dtype as Dtype,
)
from zero_flax.types import (
    Initializer as Initializer,
)
from zero_flax.types import (
    JaxArray as _JaxArray,
)
from zero_flax.types import (
    MaxFun as MaxFun,
)
from zero_flax.types import (
    PaddingLike as PaddingLike,
)
from zero_flax.types import (
    PrecisionLike as PrecisionLike,
)
from zero_flax.types import (
    Shape as Shape,
)
from zero_flax.types import (
    Size as Size,
)
from zero_flax.types import (
    ensure_array as _ensure_array,
)


def hard_swish(x: _ArrayLike) -> _JaxArray:
    """Compute the hard swish activation function ``x * relu6(x + 3) / 6``.

    Args:
        x: Input array.

    Returns:
        Hard-swish activated ``JaxArray``.
    """
    arr = _ensure_array(x)
    res: _JaxArray = arr * relu6(arr + 3.0) / 6.0
    return res


def hard_silu(x: _ArrayLike) -> _JaxArray:
    """Compute the hard SiLU activation function (alias for ``hard_swish``).

    Args:
        x: Input array.

    Returns:
        Hard-SiLU activated ``JaxArray``.
    """
    return hard_swish(x)


__all__ = [
    "All",
    "AnyFilter",
    "Array",
    "Axes",
    "Axis",
    "AxisName",
    "BatchNorm",
    "BatchStat",
    "Bidirectional",
    "Cache",
    "Conv",
    "ConvTranspose",
    "DType",
    "Dense",
    "Dict",
    "DotGeneralT",
    "Dropout",
    "Dtype",
    "Einsum",
    "Embed",
    "Everything",
    "Filter",
    "GRUCell",
    "GraphDef",
    "GroupNorm",
    "Initializer",
    "InstanceNorm",
    "Intermediate",
    "Jit",
    "LSTMCell",
    "LayerNorm",
    "Linear",
    "LinearGeneral",
    "List",
    "LoRA",
    "LoRALinear",
    "LoRAParam",
    "MaxFun",
    "Module",
    "MultiHeadAttention",
    "MultiHeadDotProductAttention",
    "Not",
    "Nothing",
    "OfType",
    "OptimizedLSTMCell",
    "PReLU",
    "PaddingLike",
    "Param",
    "PathContains",
    "Pmap",
    "PrecisionLike",
    "RMSNorm",
    "RNN",
    "RNNCellBase",
    "Remat",
    "Rng",
    "RngCount",
    "RngKey",
    "RngState",
    "RngStream",
    "Rngs",
    "RoPE",
    "Scan",
    "Sequential",
    "Shape",
    "SimpleCell",
    "Size",
    "SpectralNorm",
    "State",
    "Variable",
    "Vmap",
    "WeightNorm",
    "celu",
    "dot_product_attention",
    "elu",
    "filterlib",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "initializers",
    "iter_graph",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "merge",
    "one_hot",
    "relu",
    "relu6",
    "rnglib",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "split",
    "standardize",
    "state",
    "swish",
    "update",
    "variables",
]
