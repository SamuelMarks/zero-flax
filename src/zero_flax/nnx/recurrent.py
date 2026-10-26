"""Recurrent neural network cells and sequence unrolling wrappers for zero-flax NNX.

This module implements `RNNCellBase`, `SimpleCell`, `GRUCell`, `LSTMCell`,
`OptimizedLSTMCell`, `RNN`, and `Bidirectional` with exact `flax_nnx_v0.12.10`
parameter signatures and mathematical forward-pass execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Tuple, Union

from typing_extensions import TypeAlias
from zero_jax import numpy as jnp

from zero_flax.nnx.filterlib import FilterLiteral
from zero_flax.nnx.linear import Linear
from zero_flax.nnx.module import Module
from zero_flax.nnx.rnglib import Rngs, RngStream, resolve_rng_key
from zero_flax.types import (
    EMPTY_METADATA,
    ActivationFn,
    ArrayLike,
    DType,
    InitializerLike,
    JaxArray,
    MetadataMapping,
    MetadataValue,
    PromoteDtypeFn,
    default_kernel_init,
    default_promote_dtype,
    ensure_array,
    orthogonal_init,
    resolve_activation,
    resolve_initializer,
    sanitize_dtype_str,
    sigmoid_fn,
    tanh_fn,
    zeros_init,
)


def _assert_cell_axis_dim(tensor: JaxArray, axis: int, expected: int) -> None:
    """Validate that ``tensor`` has size ``expected`` along ``axis`` via ``zero_chex``.

    Args:
        tensor: Input or carry ``JaxArray`` to validate.
        axis: Axis index to check.
        expected: Expected dimension size along ``axis``.
    """
    from zero_chex.assertions import assert_axis_dimension

    assert_axis_dimension(tensor, axis, expected)


CarryType: TypeAlias = Union[JaxArray, Tuple[JaxArray, JaxArray]]


def _default_concat_merge(fwd: JaxArray, bwd: JaxArray) -> JaxArray:
    r"""Concatenate forward and backward RNN sequence outputs along the last axis.

    Args:
        fwd: Forward RNN output array.
        bwd: Backward RNN output array.

    Returns:
        Concatenated ``JaxArray`` along ``axis=-1``.
    """
    res: JaxArray = jnp.concatenate([fwd, bwd], axis=-1)
    return res


class RNNCellBase(Module):
    r"""Abstract base class for all recurrent neural network cells in NNX.

    Attributes:
        in_features: Input feature dimension size.
        hidden_features: Hidden carry dimension size.
        num_feature_axes: Number of feature axes consumed per time step (``1``).
    """

    in_features: int
    hidden_features: int
    num_feature_axes: int

    def __init__(self, *args: MetadataValue, **kwargs: MetadataValue) -> None:
        r"""Initialize the base RNNCellBase class.

        Args:
            *args: Optional positional configuration metadata values.
            **kwargs: Optional keyword configuration metadata values.
        """
        super().__init__(**kwargs)
        _ = args
        self.in_features = 0
        self.hidden_features = 0
        self.num_feature_axes = 1

    def initialize_carry(
        self,
        input_shape: Sequence[int],
        rngs: Optional[Union[Rngs, RngStream, bool]] = None,
    ) -> CarryType:
        r"""Initialize the hidden carry state for a given ``input_shape``.

        Args:
            input_shape: Shape ``(*batch_dims, in_features)`` of a single time-step input.
            rngs: Optional RNG stream(s) for carry initialization.

        Returns:
            Initialized carry ``JaxArray`` or ``(c, h)`` tuple of ``JaxArray``s.
        """
        _ = rngs
        batch_dims = tuple(int(d) for d in input_shape[:-1])
        return jnp.zeros((*batch_dims, max(1, self.hidden_features)))

    def __call__(
        self,
        carry: CarryType,
        inputs: ArrayLike,
    ) -> Tuple[CarryType, JaxArray]:
        r"""Execute a single recurrent time step given ``carry`` and ``inputs``.

        Args:
            carry: Current hidden state tensor (or ``(c, h)`` tuple for LSTM cells).
            inputs: Current time-step input tensor of shape ``(*batch, in_features)``.

        Returns:
            Pair ``(new_carry, output)``.
        """
        x = ensure_array(inputs)
        if isinstance(carry, tuple):
            return carry, x
        return ensure_array(carry), x


class SimpleCell(RNNCellBase):
    r"""Simple Elman recurrent cell ($h' = \phi(W_{ih} x + W_{hh} h + b)$).

    Attributes:
        in_features: Number of input features.
        hidden_features: Number of hidden state features.
        residual: Whether to add the previous carry ``h`` to the activated state.
        i2h: Input-to-hidden ``Linear`` projection (without bias).
        h2h: Hidden-to-hidden ``Linear`` projection (with bias).
    """

    in_features: int
    hidden_features: int
    dtype: Optional[DType]
    param_dtype: DType
    carry_init: InitializerLike
    residual: bool
    activation_fn: ActivationFn
    promote_dtype: PromoteDtypeFn
    keep_rngs: bool
    rngs: Optional[Union[Rngs, RngStream]]
    i2h: Linear
    h2h: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        carry_init: InitializerLike = zeros_init,
        residual: bool = False,
        activation_fn: ActivationFn = tanh_fn,
        kernel_init: InitializerLike = default_kernel_init,
        recurrent_kernel_init: InitializerLike = orthogonal_init,
        bias_init: InitializerLike = zeros_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        keep_rngs: bool = False,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        recurrent_kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize a SimpleCell recurrent module.

        Args:
            in_features: Number of input features per time step.
            hidden_features: Number of hidden carry features.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            carry_init: Initializer for the initial carry state.
            residual: Whether to add a residual connection $h' \leftarrow h' + h$.
            activation_fn: Nonlinear activation function (defaults to ``tanh``).
            kernel_init: Initializer for input-to-hidden weights.
            recurrent_kernel_init: Initializer for hidden-to-hidden weights.
            bias_init: Initializer for hidden bias.
            promote_dtype: Dtype promotion callable.
            keep_rngs: Whether to retain RNG streams on the cell instance.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata mapping for input kernel.
            recurrent_kernel_metadata: Metadata mapping for recurrent kernel.
            bias_metadata: Metadata mapping for bias.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.residual = bool(residual)
        self.activation_fn = activation_fn
        self.promote_dtype = promote_dtype
        self.keep_rngs = bool(keep_rngs)
        if self.keep_rngs and isinstance(rngs, (Rngs, RngStream)):
            self.rngs = self.register_child("rngs", rngs)
        else:
            self.rngs = None

        init_rngs: Union[Rngs, RngStream] = rngs if rngs is not None else Rngs(0)
        self.i2h = self.register_child(
            "i2h",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
            ),
        )
        self.h2h = self.register_child(
            "h2h",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.finish_init()

    def initialize_carry(
        self,
        input_shape: Sequence[int],
        rngs: Optional[Union[Rngs, RngStream, bool]] = None,
    ) -> JaxArray:
        r"""Initialize the carry tensor of shape ``(*input_shape[:-1], hidden_features)``.

        Args:
            input_shape: Single-step input shape ``(*batch, in_features)``.
            rngs: Optional RNG stream(s).

        Returns:
            Initialized carry ``JaxArray``.
        """
        c_fn = resolve_initializer(self.carry_init, zeros_init)
        eff_rngs = rngs if not isinstance(rngs, bool) and rngs is not None else self.rngs
        c_key = resolve_rng_key(eff_rngs, "carry", fold_index=0)
        batch_dims = tuple(int(d) for d in input_shape[:-1])
        return c_fn(c_key, (*batch_dims, self.hidden_features), self.param_dtype)

    def __call__(
        self,
        carry: CarryType,
        inputs: ArrayLike,
    ) -> Tuple[JaxArray, JaxArray]:
        r"""Compute one time step of the SimpleCell.

        Args:
            carry: Previous hidden state ``JaxArray`` of shape ``(*batch, hidden_features)``.
            inputs: Current time-step input ``ArrayLike`` of shape ``(*batch, in_features)``.

        Returns:
            Pair ``(new_h, new_h)``.
        """
        h = carry[0] if isinstance(carry, tuple) else ensure_array(carry)
        x = ensure_array(inputs)
        _assert_cell_axis_dim(h, -1, self.hidden_features)
        _assert_cell_axis_dim(x, -1, self.in_features)
        act = resolve_activation(self.activation_fn)
        new_h = act(self.i2h(x) + self.h2h(h))
        if self.residual:
            new_h = new_h + h
        return new_h, new_h


class GRUCell(RNNCellBase):
    r"""Gated Recurrent Unit (`GRUCell`) computing reset ($r$), update ($z$), and candidate ($	ilde{h}$) gates.

    Attributes:
        in_features: Number of input features.
        hidden_features: Number of hidden state features.
        gate_fn: Gate activation callable (defaults to ``sigmoid``).
        activation_fn: Candidate state activation callable (defaults to ``tanh``).
        ir: Input-to-reset projection.
        iz: Input-to-update projection.
        in_proj: Input-to-candidate projection.
        hr: Hidden-to-reset projection.
        hz: Hidden-to-update projection.
        hn: Hidden-to-candidate projection.
    """

    in_features: int
    hidden_features: int
    gate_fn: ActivationFn
    activation_fn: ActivationFn
    dtype: Optional[DType]
    param_dtype: DType
    keep_rngs: bool
    rngs: Optional[Union[Rngs, RngStream]]
    promote_dtype: PromoteDtypeFn
    carry_init: InitializerLike
    ir: Linear
    iz: Linear
    in_proj: Linear
    hr: Linear
    hz: Linear
    hn: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: ActivationFn = sigmoid_fn,
        activation_fn: ActivationFn = tanh_fn,
        kernel_init: InitializerLike = default_kernel_init,
        recurrent_kernel_init: InitializerLike = orthogonal_init,
        bias_init: InitializerLike = zeros_init,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        carry_init: InitializerLike = zeros_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        keep_rngs: bool = False,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        recurrent_kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize a GRUCell module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden state features.
            gate_fn: Activation function for reset and update gates (defaults to ``sigmoid``).
            activation_fn: Activation function for the candidate state (defaults to ``tanh``).
            kernel_init: Initializer for input-to-hidden weights.
            recurrent_kernel_init: Initializer for hidden-to-hidden weights.
            bias_init: Initializer for bias vectors.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            carry_init: Initializer for the initial carry state.
            promote_dtype: Dtype promotion callable.
            keep_rngs: Whether to retain RNG streams.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata mapping for input kernels.
            recurrent_kernel_metadata: Metadata mapping for recurrent kernels.
            bias_metadata: Metadata mapping for biases.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.promote_dtype = promote_dtype
        self.keep_rngs = bool(keep_rngs)
        if self.keep_rngs and isinstance(rngs, (Rngs, RngStream)):
            self.rngs = self.register_child("rngs", rngs)
        else:
            self.rngs = None

        init_rngs: Union[Rngs, RngStream] = rngs if rngs is not None else Rngs(0)
        self.ir = self.register_child(
            "ir",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.iz = self.register_child(
            "iz",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.in_proj = self.register_child(
            "in_proj",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.hr = self.register_child(
            "hr",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
            ),
        )
        self.hz = self.register_child(
            "hz",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
            ),
        )
        self.hn = self.register_child(
            "hn",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.finish_init()

    def initialize_carry(
        self,
        input_shape: Sequence[int],
        rngs: Optional[Union[Rngs, RngStream, bool]] = None,
    ) -> JaxArray:
        r"""Initialize the GRU carry tensor of shape ``(*input_shape[:-1], hidden_features)``.

        Args:
            input_shape: Single-step input shape ``(*batch, in_features)``.
            rngs: Optional RNG stream(s).

        Returns:
            Initialized carry ``JaxArray``.
        """
        c_fn = resolve_initializer(self.carry_init, zeros_init)
        eff_rngs = rngs if not isinstance(rngs, bool) and rngs is not None else self.rngs
        c_key = resolve_rng_key(eff_rngs, "carry", fold_index=0)
        batch_dims = tuple(int(d) for d in input_shape[:-1])
        return c_fn(c_key, (*batch_dims, self.hidden_features), self.param_dtype)

    def __call__(
        self,
        carry: CarryType,
        inputs: ArrayLike,
    ) -> Tuple[JaxArray, JaxArray]:
        r"""Compute one time step of the GRUCell.

        Args:
            carry: Previous hidden state ``JaxArray`` of shape ``(*batch, hidden_features)``.
            inputs: Current time-step input ``ArrayLike`` of shape ``(*batch, in_features)``.

        Returns:
            Pair ``(new_h, new_h)``.
        """
        h = carry[0] if isinstance(carry, tuple) else ensure_array(carry)
        x = ensure_array(inputs)
        _assert_cell_axis_dim(h, -1, self.hidden_features)
        _assert_cell_axis_dim(x, -1, self.in_features)
        g_fn = resolve_activation(self.gate_fn)
        act_fn = resolve_activation(self.activation_fn)

        r = g_fn(self.ir(x) + self.hr(h))
        z = g_fn(self.iz(x) + self.hz(h))
        n = act_fn(self.in_proj(x) + r * self.hn(h))
        new_h: JaxArray = (1.0 - z) * n + z * h
        return new_h, new_h


class LSTMCell(RNNCellBase):
    r"""Long Short-Term Memory (`LSTMCell`) computing input ($i$), forget ($f$), cell ($g$), and output ($o$) gates.

    Attributes:
        in_features: Number of input features.
        hidden_features: Number of hidden state and cell state features.
        ii: Input-to-input-gate projection.
        if_proj: Input-to-forget-gate projection.
        ig: Input-to-cell-gate projection.
        io: Input-to-output-gate projection.
        hi: Hidden-to-input-gate projection.
        hf: Hidden-to-forget-gate projection.
        hg: Hidden-to-cell-gate projection.
        ho: Hidden-to-output-gate projection.
    """

    in_features: int
    hidden_features: int
    gate_fn: ActivationFn
    activation_fn: ActivationFn
    dtype: Optional[DType]
    param_dtype: DType
    carry_init: InitializerLike
    promote_dtype: PromoteDtypeFn
    keep_rngs: bool
    rngs: Optional[Union[Rngs, RngStream]]
    ii: Linear
    if_proj: Linear
    ig: Linear
    io: Linear
    hi: Linear
    hf: Linear
    hg: Linear
    ho: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: ActivationFn = sigmoid_fn,
        activation_fn: ActivationFn = tanh_fn,
        kernel_init: InitializerLike = default_kernel_init,
        recurrent_kernel_init: InitializerLike = orthogonal_init,
        bias_init: InitializerLike = zeros_init,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        carry_init: InitializerLike = zeros_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        keep_rngs: bool = False,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        recurrent_kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize an LSTMCell module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden and cell features.
            gate_fn: Gate activation function (defaults to ``sigmoid``).
            activation_fn: Cell/output activation function (defaults to ``tanh``).
            kernel_init: Initializer for input-to-gate weights.
            recurrent_kernel_init: Initializer for hidden-to-gate weights.
            bias_init: Initializer for gate biases.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            carry_init: Initializer for initial ``(c, h)`` carry states.
            promote_dtype: Dtype promotion callable.
            keep_rngs: Whether to retain RNG streams.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata mapping for input kernels.
            recurrent_kernel_metadata: Metadata mapping for recurrent kernels.
            bias_metadata: Metadata mapping for biases.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.promote_dtype = promote_dtype
        self.keep_rngs = bool(keep_rngs)
        if self.keep_rngs and isinstance(rngs, (Rngs, RngStream)):
            self.rngs = self.register_child("rngs", rngs)
        else:
            self.rngs = None

        init_rngs: Union[Rngs, RngStream] = rngs if rngs is not None else Rngs(0)
        self.ii = self.register_child(
            "ii",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
            ),
        )
        self.if_proj = self.register_child(
            "if_proj",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
            ),
        )
        self.ig = self.register_child(
            "ig",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
            ),
        )
        self.io = self.register_child(
            "io",
            Linear(
                self.in_features,
                self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
            ),
        )
        self.hi = self.register_child(
            "hi",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.hf = self.register_child(
            "hf",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.hg = self.register_child(
            "hg",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.ho = self.register_child(
            "ho",
            Linear(
                self.hidden_features,
                self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.finish_init()

    def initialize_carry(
        self,
        input_shape: Sequence[int],
        rngs: Optional[Union[Rngs, RngStream, bool]] = None,
    ) -> Tuple[JaxArray, JaxArray]:
        r"""Initialize the ``(c, h)`` carry tuple for ``input_shape``.

        Args:
            input_shape: Single-step input shape ``(*batch, in_features)``.
            rngs: Optional RNG stream(s).

        Returns:
            Pair ``(c, h)`` of initialized ``JaxArray`` tensors.
        """
        c_fn = resolve_initializer(self.carry_init, zeros_init)
        eff_rngs = rngs if not isinstance(rngs, bool) and rngs is not None else self.rngs
        key1 = resolve_rng_key(eff_rngs, "carry", fold_index=0)
        key2 = resolve_rng_key(eff_rngs, "carry", fold_index=1)
        batch_dims = tuple(int(d) for d in input_shape[:-1])
        target_shape = (*batch_dims, self.hidden_features)
        return (
            c_fn(key1, target_shape, self.param_dtype),
            c_fn(key2, target_shape, self.param_dtype),
        )

    def __call__(
        self,
        carry: CarryType,
        inputs: ArrayLike,
    ) -> Tuple[Tuple[JaxArray, JaxArray], JaxArray]:
        r"""Compute one time step of the LSTMCell.

        Args:
            carry: Tuple ``(c, h)`` of cell and hidden state tensors.
            inputs: Current time-step input tensor of shape ``(*batch, in_features)``.

        Returns:
            Pair ``((new_c, new_h), new_h)``.
        """
        if isinstance(carry, tuple):
            c, h = ensure_array(carry[0]), ensure_array(carry[1])
        else:
            c = ensure_array(carry)
            h = ensure_array(carry)
        x = ensure_array(inputs)
        _assert_cell_axis_dim(c, -1, self.hidden_features)
        _assert_cell_axis_dim(h, -1, self.hidden_features)
        _assert_cell_axis_dim(x, -1, self.in_features)
        g_fn = resolve_activation(self.gate_fn)
        act_fn = resolve_activation(self.activation_fn)

        i = g_fn(self.ii(x) + self.hi(h))
        f = g_fn(self.if_proj(x) + self.hf(h))
        g = act_fn(self.ig(x) + self.hg(h))
        o = g_fn(self.io(x) + self.ho(h))
        new_c: JaxArray = f * c + i * g
        new_h: JaxArray = o * act_fn(new_c)
        return (new_c, new_h), new_h


class OptimizedLSTMCell(RNNCellBase):
    r"""Fused-projection Long Short-Term Memory (`OptimizedLSTMCell`) concatenating all 4 gates.

    Attributes:
        in_features: Number of input features.
        hidden_features: Number of hidden and cell features.
        dense_i: Fused input-to-gates ``Linear`` layer of output size ``4 * hidden_features``.
        dense_h: Fused hidden-to-gates ``Linear`` layer of output size ``4 * hidden_features``.
    """

    in_features: int
    hidden_features: int
    gate_fn: ActivationFn
    activation_fn: ActivationFn
    dtype: Optional[DType]
    param_dtype: DType
    keep_rngs: bool
    rngs: Optional[Union[Rngs, RngStream]]
    promote_dtype: PromoteDtypeFn
    carry_init: InitializerLike
    dense_i: Linear
    dense_h: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: ActivationFn = sigmoid_fn,
        activation_fn: ActivationFn = tanh_fn,
        kernel_init: InitializerLike = default_kernel_init,
        recurrent_kernel_init: InitializerLike = orthogonal_init,
        bias_init: InitializerLike = zeros_init,
        dtype: Optional[DType] = None,
        param_dtype: DType = "float32",
        carry_init: InitializerLike = zeros_init,
        promote_dtype: PromoteDtypeFn = default_promote_dtype,
        keep_rngs: bool = False,
        rngs: Optional[Union[Rngs, RngStream]] = None,
        kernel_metadata: MetadataMapping = EMPTY_METADATA,
        recurrent_kernel_metadata: MetadataMapping = EMPTY_METADATA,
        bias_metadata: MetadataMapping = EMPTY_METADATA,
    ) -> None:
        r"""Initialize an OptimizedLSTMCell module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden and cell features.
            gate_fn: Gate activation function (defaults to ``sigmoid``).
            activation_fn: Cell/output activation function (defaults to ``tanh``).
            kernel_init: Initializer for fused input weights.
            recurrent_kernel_init: Initializer for fused recurrent weights.
            bias_init: Initializer for fused gate biases.
            dtype: Computation data type.
            param_dtype: Parameter initialization data type.
            carry_init: Initializer for initial ``(c, h)`` carry states.
            promote_dtype: Dtype promotion callable.
            keep_rngs: Whether to retain RNG streams.
            rngs: Random number generator stream(s).
            kernel_metadata: Metadata mapping for ``dense_i`` kernel.
            recurrent_kernel_metadata: Metadata mapping for ``dense_h`` kernel.
            bias_metadata: Metadata mapping for ``dense_h`` bias.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.promote_dtype = promote_dtype
        self.keep_rngs = bool(keep_rngs)
        if self.keep_rngs and isinstance(rngs, (Rngs, RngStream)):
            self.rngs = self.register_child("rngs", rngs)
        else:
            self.rngs = None

        init_rngs: Union[Rngs, RngStream] = rngs if rngs is not None else Rngs(0)
        self.dense_i = self.register_child(
            "dense_i",
            Linear(
                self.in_features,
                4 * self.hidden_features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=kernel_metadata,
            ),
        )
        self.dense_h = self.register_child(
            "dense_h",
            Linear(
                self.hidden_features,
                4 * self.hidden_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
                promote_dtype=promote_dtype,
                rngs=init_rngs,
                kernel_metadata=recurrent_kernel_metadata,
                bias_metadata=bias_metadata,
            ),
        )
        self.finish_init()

    def initialize_carry(
        self,
        input_shape: Sequence[int],
        rngs: Optional[Union[Rngs, RngStream, bool]] = None,
    ) -> Tuple[JaxArray, JaxArray]:
        r"""Initialize the ``(c, h)`` carry tuple for ``input_shape``.

        Args:
            input_shape: Single-step input shape ``(*batch, in_features)``.
            rngs: Optional RNG stream(s).

        Returns:
            Pair ``(c, h)`` of initialized ``JaxArray`` tensors.
        """
        c_fn = resolve_initializer(self.carry_init, zeros_init)
        eff_rngs = rngs if not isinstance(rngs, bool) and rngs is not None else self.rngs
        key1 = resolve_rng_key(eff_rngs, "carry", fold_index=0)
        key2 = resolve_rng_key(eff_rngs, "carry", fold_index=1)
        batch_dims = tuple(int(d) for d in input_shape[:-1])
        target_shape = (*batch_dims, self.hidden_features)
        return (
            c_fn(key1, target_shape, self.param_dtype),
            c_fn(key2, target_shape, self.param_dtype),
        )

    def __call__(
        self,
        carry: CarryType,
        inputs: ArrayLike,
    ) -> Tuple[Tuple[JaxArray, JaxArray], JaxArray]:
        r"""Compute one fused time step of the OptimizedLSTMCell.

        Args:
            carry: Tuple ``(c, h)`` of cell and hidden state tensors.
            inputs: Current time-step input tensor of shape ``(*batch, in_features)``.

        Returns:
            Pair ``((new_c, new_h), new_h)``.
        """
        if isinstance(carry, tuple):
            c, h = ensure_array(carry[0]), ensure_array(carry[1])
        else:
            c = ensure_array(carry)
            h = ensure_array(carry)
        x = ensure_array(inputs)
        _assert_cell_axis_dim(c, -1, self.hidden_features)
        _assert_cell_axis_dim(h, -1, self.hidden_features)
        _assert_cell_axis_dim(x, -1, self.in_features)
        g_fn = resolve_activation(self.gate_fn)
        act_fn = resolve_activation(self.activation_fn)

        gates = self.dense_i(x) + self.dense_h(h)
        hid = self.hidden_features
        i = g_fn(gates[..., 0:hid])
        f = g_fn(gates[..., hid : 2 * hid])
        g = act_fn(gates[..., 2 * hid : 3 * hid])
        o = g_fn(gates[..., 3 * hid : 4 * hid])
        new_c: JaxArray = f * c + i * g
        new_h: JaxArray = o * act_fn(new_c)
        return (new_c, new_h), new_h


def _select_masked_carry(
    mask_cond: JaxArray,
    new_carry: CarryType,
    old_carry: CarryType,
) -> CarryType:
    """Select ``new_carry`` where ``mask_cond`` is ``True`` and ``old_carry`` where ``False``.

    Args:
        mask_cond: Boolean mask broadcastable to carry batch dimensions.
        new_carry: Candidate updated carry state.
        old_carry: Previous carry state to retain on padded steps.

    Returns:
        Masked carry state matching the structure of ``new_carry``.
    """
    if isinstance(new_carry, tuple) and isinstance(old_carry, tuple):
        return (
            jnp.where(mask_cond, new_carry[0], old_carry[0]),
            jnp.where(mask_cond, new_carry[1], old_carry[1]),
        )
    nc = new_carry[0] if isinstance(new_carry, tuple) else new_carry
    oc = old_carry[0] if isinstance(old_carry, tuple) else old_carry
    return jnp.where(mask_cond, nc, oc)


class RNN(Module):
    r"""Recurrent sequence wrapper unrolling an ``RNNCellBase`` across the time dimension.

    Attributes:
        cell: Wrapped ``RNNCellBase`` instance.
        time_major: If ``True``, sequence dimension is axis ``0``; otherwise axis ``1``.
        return_carry: Whether ``__call__`` returns ``(final_carry, outputs)`` instead of ``outputs``.
        reverse: Whether to unroll the input sequence in reverse time order.
        keep_order: When ``reverse=True``, whether to flip outputs back to original time order.
        unroll: Loop unrolling factor.
    """

    cell: RNNCellBase
    time_major: bool
    return_carry: bool
    reverse: bool
    keep_order: bool
    unroll: int
    rngs: Union[Rngs, RngStream, bool, None]
    broadcast_rngs: FilterLiteral
    state_axes: Optional[Mapping[str, Optional[int]]]

    def __init__(
        self,
        cell: RNNCellBase,
        *,
        time_major: bool = False,
        return_carry: bool = False,
        reverse: bool = False,
        keep_order: bool = False,
        unroll: int = 1,
        rngs: Union[Rngs, RngStream, bool, None] = True,
        broadcast_rngs: FilterLiteral = None,
        state_axes: Optional[Mapping[str, Optional[int]]] = None,
    ) -> None:
        r"""Initialize an RNN sequence unrolling wrapper.

        Args:
            cell: Recurrent cell instance inheriting from ``RNNCellBase``.
            time_major: Whether the time axis is the first dimension (axis ``0``).
            return_carry: Whether to return ``(carry, outputs)`` from ``__call__``.
            reverse: Whether to iterate over time steps from end to start.
            keep_order: Whether to restore chronological order of outputs when ``reverse=True``.
            unroll: Loop unroll factor. Defaults to ``1``.
            rngs: Optional RNG stream(s) or boolean flag.
            broadcast_rngs: Optional filter specifying RNG streams to broadcast across time steps.
            state_axes: Optional mapping of state variable axis specifications.
        """
        super().__init__()
        self.cell = self.register_child("cell", cell)
        self.time_major = bool(time_major)
        self.return_carry = bool(return_carry)
        self.reverse = bool(reverse)
        self.keep_order = bool(keep_order)
        self.unroll = int(unroll)
        self.rngs = rngs
        self.broadcast_rngs = broadcast_rngs
        self.state_axes = state_axes
        self.finish_init()

    def __call__(
        self,
        inputs: ArrayLike,
        *,
        initial_carry: Optional[CarryType] = None,
        seq_lengths: Optional[ArrayLike] = None,
        return_carry: Optional[bool] = None,
        time_major: Optional[bool] = None,
        reverse: Optional[bool] = None,
        keep_order: Optional[bool] = None,
    ) -> Union[JaxArray, Tuple[CarryType, JaxArray]]:
        r"""Unroll ``self.cell`` over the sequence dimension of ``inputs``.

        Args:
            inputs: Sequence tensor of shape ``(batch, time, features)`` (or ``(time, batch, features)`` if ``time_major``).
            initial_carry: Optional explicit starting carry state.
            seq_lengths: Optional integer array of valid sequence lengths per batch element.
            return_carry: Optional override for ``self.return_carry``.
            time_major: Optional override for ``self.time_major``.
            reverse: Optional override for ``self.reverse``.
            keep_order: Optional override for ``self.keep_order``.

        Returns:
            Output sequence ``JaxArray``, or ``(final_carry, output_sequence)`` if ``return_carry`` is ``True``.
        """
        x = ensure_array(inputs)
        tm = self.time_major if time_major is None else bool(time_major)
        rev = self.reverse if reverse is None else bool(reverse)
        ko = self.keep_order if keep_order is None else bool(keep_order)
        ret_c = self.return_carry if return_carry is None else bool(return_carry)

        time_axis = 0 if tm else 1
        seq_len = x.shape[time_axis]
        step_shape = (*x.shape[1:],) if tm else (x.shape[0], *x.shape[2:])
        carry: CarryType = (
            initial_carry
            if initial_carry is not None
            else self.cell.initialize_carry(step_shape, rngs=self.rngs)
        )

        if seq_len == 0:
            out_dt = sanitize_dtype_str(str(x.dtype).lower())
            empty_shape = (
                (0, *step_shape[:-1], self.cell.hidden_features)
                if tm
                else (step_shape[0], 0, *step_shape[1:-1], self.cell.hidden_features)
            )
            empty_seq: JaxArray = jnp.zeros(empty_shape, dtype=out_dt)
            if ret_c:
                return carry, empty_seq
            return empty_seq

        seq_len_arr: Optional[JaxArray] = (
            ensure_array(seq_lengths, dtype="int32") if seq_lengths is not None else None
        )
        indices = list(range(seq_len - 1, -1, -1)) if rev else list(range(seq_len))
        outputs: list[JaxArray] = []
        for t_idx in indices:
            step_x = x[t_idx] if tm else x[:, t_idx]
            prev_carry = carry
            next_carry, step_y = self.cell(carry, step_x)
            if seq_len_arr is not None:
                valid_mask = jnp.reshape(
                    t_idx < seq_len_arr,
                    (-1,) + (1,) * max(0, step_y.ndim - 1),
                )
                carry = _select_masked_carry(valid_mask, next_carry, prev_carry)
                step_y = jnp.where(valid_mask, step_y, jnp.zeros_like(step_y))
            else:
                carry = next_carry
            outputs.append(jnp.expand_dims(step_y, axis=time_axis))

        if rev and ko:
            outputs.reverse()

        stacked: JaxArray = jnp.concatenate(outputs, axis=time_axis)
        if ret_c:
            return carry, stacked
        return stacked


class Bidirectional(Module):
    r"""Bidirectional recurrent wrapper combining a forward RNN and a backward RNN.

    Attributes:
        forward_rnn: Forward ``RNN`` wrapper module.
        backward_rnn: Backward ``RNN`` wrapper module.
        merge_fn: Callable combining forward and backward output sequences.
        time_major: Whether the sequence axis is axis ``0``.
        return_carry: Whether to return ``((fwd_carry, bwd_carry), merged_outputs)``.
    """

    forward_rnn: RNN
    backward_rnn: RNN
    merge_fn: Callable[[JaxArray, JaxArray], JaxArray]
    time_major: bool
    return_carry: bool

    def __init__(
        self,
        forward_rnn: Union[RNN, RNNCellBase],
        backward_rnn: Union[RNN, RNNCellBase],
        *,
        merge_fn: Callable[[JaxArray, JaxArray], JaxArray] = _default_concat_merge,
        time_major: bool = False,
        return_carry: bool = False,
        rngs: Union[Rngs, RngStream, bool] = True,
    ) -> None:
        r"""Initialize a Bidirectional recurrent wrapper.

        Args:
            forward_rnn: Forward ``RNN`` or ``RNNCellBase`` module.
            backward_rnn: Backward ``RNN`` or ``RNNCellBase`` module.
            merge_fn: Function taking ``(forward_out, backward_out)`` and returning a merged tensor.
            time_major: Whether the sequence dimension is axis ``0``.
            return_carry: Whether to return final carries alongside the merged sequence output.
            rngs: Optional RNG stream(s) or boolean flag.
        """
        super().__init__()
        fwd_mod = (
            forward_rnn
            if isinstance(forward_rnn, RNN)
            else RNN(forward_rnn, time_major=time_major, rngs=rngs)
        )
        bwd_mod = (
            backward_rnn
            if isinstance(backward_rnn, RNN)
            else RNN(
                backward_rnn,
                time_major=time_major,
                reverse=True,
                keep_order=True,
                rngs=rngs,
            )
        )
        self.forward_rnn = self.register_child("forward_rnn", fwd_mod)
        self.backward_rnn = self.register_child("backward_rnn", bwd_mod)
        self.merge_fn = merge_fn
        self.time_major = bool(time_major)
        self.return_carry = bool(return_carry)
        self.finish_init()

    def __call__(
        self,
        inputs: ArrayLike,
        *,
        initial_carry: Optional[Tuple[CarryType, CarryType]] = None,
        seq_lengths: Optional[ArrayLike] = None,
        return_carry: Optional[bool] = None,
        time_major: Optional[bool] = None,
        reverse: Optional[bool] = None,
        keep_order: Optional[bool] = None,
    ) -> Union[JaxArray, Tuple[Tuple[CarryType, CarryType], JaxArray]]:
        r"""Run forward and backward RNN passes on ``inputs`` and merge their outputs.

        Args:
            inputs: Sequence input tensor.
            initial_carry: Optional tuple ``(fwd_initial_carry, bwd_initial_carry)``.
            seq_lengths: Optional integer array of valid sequence lengths per batch element.
            return_carry: Optional override for ``self.return_carry``.
            time_major: Optional override for ``self.time_major``.
            reverse: Optional override for backward RNN reverse flag.
            keep_order: Optional override for backward RNN keep_order flag.

        Returns:
            Merged output ``JaxArray``, or ``((fwd_carry, bwd_carry), merged_output)`` if ``return_carry``.
        """
        tm = self.time_major if time_major is None else bool(time_major)
        ret_c = self.return_carry if return_carry is None else bool(return_carry)
        bwd_rev = self.backward_rnn.reverse if reverse is None else bool(reverse)
        bwd_ko = self.backward_rnn.keep_order if keep_order is None else bool(keep_order)
        fwd_init_c = initial_carry[0] if initial_carry is not None else None
        bwd_init_c = initial_carry[1] if initial_carry is not None else None

        fwd_res = self.forward_rnn(
            inputs,
            initial_carry=fwd_init_c,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=tm,
            reverse=self.forward_rnn.reverse,
            keep_order=self.forward_rnn.keep_order,
        )
        bwd_res = self.backward_rnn(
            inputs,
            initial_carry=bwd_init_c,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=tm,
            reverse=bwd_rev,
            keep_order=bwd_ko,
        )
        fwd_carry, fwd_out = fwd_res if isinstance(fwd_res, tuple) else (fwd_res, fwd_res)
        bwd_carry, bwd_out = bwd_res if isinstance(bwd_res, tuple) else (bwd_res, bwd_res)
        merged = self.merge_fn(ensure_array(fwd_out), ensure_array(bwd_out))
        if ret_c:
            return (fwd_carry, bwd_carry), merged
        return merged


__all__ = [
    "Bidirectional",
    "CarryType",
    "GRUCell",
    "LSTMCell",
    "OptimizedLSTMCell",
    "RNN",
    "RNNCellBase",
    "SimpleCell",
    "default_kernel_init",
    "orthogonal_init",
]
