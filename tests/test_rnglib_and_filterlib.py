"""Unit tests for `rnglib`, `filterlib`, `state`, `module`, `types`, `conv`, and `transforms` edge branches."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from zero_jax import numpy as jnp

import zero_flax.nnx as nnx
from zero_flax.nnx import filterlib, rnglib
from zero_flax.types import (
    ArrayLike,
    DimensionNumbers,
    DotGeneralT,
    DType,
    Dtype,
    Initializer,
    JaxArray,
    PaddingLike,
    PrecisionSpec,
    PRNGKeyLike,
    compute_dot_general,
    default_embed_init,
    default_promote_dtype,
    ensure_array,
    is_int_sequence,
    is_padding_pair_sequence,
    is_tensor,
    normalize_int_tuple,
    orthogonal_init,
    resolve_activation,
    resolve_initializer,
    sanitize_dtype_str,
)


def test_rnglib_branches() -> None:
    """Verify RngStream and Rngs edge cases including fork, fallback stream creation, and KeyError."""
    stream = rnglib.RngStream(None, tag="custom")
    stream_arr = rnglib.RngStream((0, 0), tag="tuple_key")
    assert stream_arr() is not None
    assert stream.fork() is not None
    assert stream.fork(2).shape[0] == 2
    assert stream.fork((2, 3)).shape == (2, 3, 2)

    rngs_no_default = rnglib.Rngs(default=None, params=42)
    assert rngs_no_default.has_stream("params")
    assert not rngs_no_default.has_stream("missing")
    with pytest.raises(KeyError, match="No RNG stream found"):
        rngs_no_default("new_stream")
    assert rnglib.resolve_rng_key(stream) is not None
    assert rnglib.resolve_rng_key(True) is not None


def test_filterlib_and_state_branches() -> None:
    """Verify filterlib.to_predicate, State.flat_state, State.flat_tuple_state, and iter_graph."""
    pred_custom = filterlib.to_predicate(lambda path, _val: "w" in path)
    assert pred_custom(("w",), nnx.Param(1))

    pred_true = filterlib.to_predicate(True)
    assert pred_true(("a",), nnx.Param(1))

    pred_false = filterlib.to_predicate(False)
    assert not pred_false(("a",), nnx.Param(1))

    pred_str = filterlib.to_predicate("kernel")
    assert pred_str(("d1", "kernel"), nnx.Param(1))

    st_kw = nnx.State(a=nnx.Param(1))
    assert "a" in st_kw

    st = nnx.State(
        {
            "d1": nnx.State(
                {
                    "sub": nnx.State({"leaf": nnx.Param(10)}),
                    "kernel": nnx.Param(1),
                    "bias": nnx.Param(2),
                }
            ),
            "top": nnx.BatchStat(3),
            "raw_int": 99,
            "empty_sub": nnx.State({"other": nnx.BatchStat(4)}),
        }
    )
    p_state, rest_state = st.split(nnx.Param)
    assert "d1" in p_state
    assert "empty_sub" not in p_state
    assert "empty_sub" in rest_state

    flat = st.flat_state()
    assert "d1.kernel" in flat
    assert "d1.bias" in flat
    assert "d1.sub.leaf" in flat
    assert "top" in flat

    flat_tuples = st.flat_tuple_state()
    assert ("d1", "kernel") in flat_tuples
    assert ("top",) in flat_tuples

    reconstructed = nnx.State.from_flat_state(flat)
    assert isinstance(reconstructed["d1"], nnx.State)
    assert "kernel" in reconstructed["d1"]

    reconstructed_tuples = nnx.State.from_flat_state(flat_tuples)
    assert isinstance(reconstructed_tuples["d1"], nnx.State)

    merged_nested = nnx.state.merge(
        nnx.State({"d1": nnx.State({"a": nnx.Param(1)})}),
        nnx.State({"d1": nnx.State({"b": nnx.Param(2)})}),
    )
    assert isinstance(merged_nested["d1"], nnx.State)
    assert "a" in merged_nested["d1"] and "b" in merged_nested["d1"]

    state_nodes = list(nnx.iter_graph(st))
    assert len(state_nodes) >= 5

    cyclic_mod = nnx.Module()
    cyclic_mod.register_child("self_ref", cyclic_mod)
    cyclic_nodes = list(nnx.iter_graph(cyclic_mod))
    assert len(cyclic_nodes) == 1


def test_types_and_conv_branches() -> None:
    """Verify helper functions in zero_flax.types and Conv/ConvTranspose padding/dilation/group branches."""
    arr = jnp.ones((2, 2))
    assert not is_tensor(None)
    assert is_tensor(Tensor([1.0, 2.0], TensorConfig((2,), "float32", "cpu")))
    assert not is_int_sequence(3)
    assert is_int_sequence((3, 4))
    assert not is_padding_pair_sequence("SAME")
    assert not is_padding_pair_sequence(())
    assert is_padding_pair_sequence(((1, 1), (2, 2)))
    assert normalize_int_tuple(None, 2, default=1) == (1, 1)
    assert normalize_int_tuple((3,), 2) == (3, 3)

    dt = Dtype("float32")
    assert repr(dt) == "Dtype('float32')"
    assert sanitize_dtype_str(None) == "float32"
    assert sanitize_dtype_str(dt) == "float32"
    assert sanitize_dtype_str(CompilerDType("float32")) == "float32"
    assert sanitize_dtype_str(bool) == "bool"
    assert sanitize_dtype_str(int) == "int32"
    assert sanitize_dtype_str(float) == "float32"
    with pytest.raises(ValueError, match="Unsupported Dtype name"):
        sanitize_dtype_str(Dtype("invalid_dt"))
    with pytest.raises(ValueError, match="Unsupported dtype string"):
        sanitize_dtype_str("invalid_dt")

    axes_single = nnx.Axes(-1)
    assert tuple(axes_single) == (-1,)
    ln_axes = nnx.LayerNorm(2, reduction_axes=axes_single)
    assert ln_axes(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    assert ensure_array([1.0, 2.0], dtype=dt).shape == (2,)
    assert ensure_array([1.0, 2.0], dtype="float32").shape == (2,)
    promoted = default_promote_dtype(None, [1, 2], inexact=True)
    assert len(promoted) == 2
    promoted_typed = default_promote_dtype([1.0, 2.0], dtype="float32")
    assert len(promoted_typed) == 1

    assert default_embed_init(None, (3, 2), dt).shape == (3, 2)
    assert orthogonal_init(None, (2, 2), dt).shape == (2, 2)

    def _two_arg_init(
        _k: PRNGKeyLike,
        s: Sequence[int],
    ) -> JaxArray:
        """Return zeros of shape ``s``.

        Args:
            _k: Unused PRNG key.
            s: Target shape.

        Returns:
            Zeros ``JaxArray``.
        """
        res: JaxArray = jnp.zeros(s)
        return res

    def _three_arg_init(
        _k: PRNGKeyLike,
        s: Sequence[int],
        _d: DType | None = None,
    ) -> JaxArray:
        """Return ones of shape ``s``.

        Args:
            _k: Unused PRNG key.
            s: Target shape.
            _d: Unused dtype.

        Returns:
            Ones ``JaxArray``.
        """
        res: JaxArray = jnp.ones(s)
        return res

    init_wrapper = Initializer(_three_arg_init)
    assert resolve_initializer(init_wrapper)((0, 0), (2, 2), "float32").shape == (2, 2)
    assert resolve_initializer(None)((0, 0), (2, 2), "float32").shape == (2, 2)
    assert resolve_initializer(_two_arg_init)((0, 0), (2, 2), dt).shape == (2, 2)
    assert resolve_initializer(_three_arg_init)((0, 0), (2, 2), dt).shape == (2, 2)

    for act_name in (
        "sigmoid",
        "tanh",
        "relu",
        "gelu",
        "silu",
        "elu",
        "celu",
        "selu",
        "softplus",
        "soft_sign",
        "hard_tanh",
        "hard_sigmoid",
        "hard_swish",
        "leaky_relu",
        "log_sigmoid",
        "relu6",
    ):
        act_fn = resolve_activation(act_name)
        assert act_fn(0.5) is not None

    with pytest.raises(ValueError, match="Unsupported activation"):
        resolve_activation("unknown_act")

    def _custom_dg(
        lhs: ArrayLike,
        rhs: ArrayLike,
        dimension_numbers: DimensionNumbers,
        precision: PrecisionSpec = None,
        preferred_element_type: DType | None = None,
    ) -> JaxArray:
        """Custom dot_general function using ``jnp.dot``.

        Args:
            lhs: Left operand.
            rhs: Right operand.
            dimension_numbers: Dimension numbers.
            precision: Precision spec.
            preferred_element_type: Preferred element type.

        Returns:
            Dot product ``JaxArray``.
        """
        _ = (dimension_numbers, precision, preferred_element_type)
        res: JaxArray = jnp.dot(ensure_array(lhs), ensure_array(rhs))
        return res

    dg_custom = DotGeneralT(_custom_dg)
    assert dg_custom(arr, arr, (((1,), (0,)), ((), ()))).shape == (2, 2)
    assert compute_dot_general(arr, arr, ((1,), (0,))).shape == (2, 2)

    lg_custom_dg = nnx.LinearGeneral(2, 2, dot_general=dg_custom)
    assert lg_custom_dg(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)
    with pytest.raises(ValueError, match="LinearGeneral input dimension"):
        lg_custom_dg(np.ones((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Linear input last dimension"):
        nnx.Linear(2, 2)(np.ones((1, 3), dtype=np.float32))

    ll_extra = nnx.LoRALinear(2, 2, base_module=nnx.Linear(2, 2), custom_flag=True)
    assert ll_extra._static_config["custom_flag"] is True
    assert ll_extra(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    lora_with_base = nnx.LoRA(2, 1, 2, base_module=nnx.Linear(2, 2), promote_dtype=None)
    assert lora_with_base(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    with pytest.raises(ValueError, match="feature_group_count must be a positive"):
        nnx.Conv(2, 2, (2,), feature_group_count=0)
    with pytest.raises(ValueError, match="in_features"):
        nnx.Conv(3, 2, (2,), feature_group_count=2)
    with pytest.raises(ValueError, match="out_features"):
        nnx.Conv(2, 3, (2,), feature_group_count=2)

    mask = np.ones((2, 1, 2), dtype=np.float32)
    c_causal = nnx.Conv(
        2,
        2,
        kernel_size=(2,),
        padding="CAUSAL",
        feature_group_count=2,
        mask=mask,
        preferred_element_type="float32",
        promote_dtype=None,
    )
    assert c_causal(np.ones((1, 4, 2), dtype=np.float32)).shape == (1, 4, 2)

    c_circ = nnx.Conv(2, 2, kernel_size=(3, 3), padding="CIRCULAR", input_dilation=(1, 2))
    assert c_circ(np.ones((1, 3, 3, 2), dtype=np.float32)).shape == (1, 3, 5, 2)
    assert c_circ(np.ones((1, 1, 1, 2), dtype=np.float32)).shape == (1, 1, 1, 2)

    c_custom_cgd = nnx.Conv(
        2,
        2,
        kernel_size=(1,),
        conv_general_dilated=lambda lhs, rhs: jnp.dot(ensure_array(lhs), ensure_array(rhs)[0]),
    )
    assert c_custom_cgd(np.ones((1, 3, 2), dtype=np.float32)).shape == (1, 3, 2)
    c_custom_cgd_nb = nnx.Conv(
        2,
        2,
        kernel_size=(1,),
        use_bias=False,
        conv_general_dilated=lambda lhs, rhs: jnp.dot(ensure_array(lhs), ensure_array(rhs)[0]),
    )
    assert c_custom_cgd_nb(np.ones((1, 3, 2), dtype=np.float32)).shape == (1, 3, 2)

    c_pairs = nnx.Conv(2, 2, kernel_size=(2,), padding=((1, 2),))
    assert c_pairs(np.ones((1, 4, 2), dtype=np.float32)).shape == (1, 6, 2)

    c_int_pad = nnx.Conv(2, 2, kernel_size=(2,), padding=1)
    assert c_int_pad(np.ones((1, 4, 2), dtype=np.float32)).shape == (1, 5, 2)

    c_seq_pad = nnx.Conv(2, 2, kernel_size=(2,), padding=PaddingLike([1]))
    assert c_seq_pad(np.ones((1, 4, 2), dtype=np.float32)).shape == (1, 5, 2)

    c_bad_pad = nnx.Conv(2, 2, kernel_size=(2,), padding="INVALID_MODE")
    with pytest.raises(ValueError, match="Unsupported convolution padding"):
        c_bad_pad(np.ones((1, 4, 2), dtype=np.float32))

    c_short = nnx.Conv(2, 2, kernel_size=(5,), padding="VALID")
    assert c_short(np.ones((1, 6, 2), dtype=np.float32)).shape == (1, 2, 2)
    with pytest.raises(ValueError, match="Conv expects input"):
        c_short(np.ones((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="Conv input channel"):
        c_short(np.ones((1, 6, 3), dtype=np.float32))

    ct_valid = nnx.ConvTranspose(
        2,
        2,
        kernel_size=(2,),
        strides=(1,),
        padding="VALID",
        kernel_dilation=(1,),
        mask=np.ones((2, 2, 2)),
        transpose_kernel=True,
        preferred_element_type="float32",
        promote_dtype=None,
    )
    assert ct_valid(np.ones((1, 3, 2), dtype=np.float32)).shape == (1, 4, 2)
    with pytest.raises(ValueError, match="ConvTranspose expects input"):
        ct_valid(np.ones((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="ConvTranspose input channel"):
        ct_valid(np.ones((1, 3, 3), dtype=np.float32))

    ct_int_pad = nnx.ConvTranspose(2, 2, kernel_size=(2,), strides=(1,), padding=1, use_bias=False)
    assert ct_int_pad(np.ones((1, 3, 2), dtype=np.float32)).shape == (1, 4, 2)

    ct_bad_pad = nnx.ConvTranspose(2, 2, kernel_size=(2,), padding="INVALID_MODE")
    with pytest.raises(ValueError, match="Unsupported transposed convolution padding"):
        ct_bad_pad(np.ones((1, 3, 2), dtype=np.float32))


def test_transforms_and_module_edge_branches() -> None:
    """Verify Vmap, Scan, Pmap, Einsum, LoRA, and Module edge branches."""
    ein_custom = nnx.Einsum(
        "ab,bc->ac",
        (2, 2),
        einsum_op=lambda eq, a, b: jnp.einsum(eq, a, b),
    )
    assert ein_custom(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    lora_plain = nnx.LoRA(2, 1, 2)
    assert lora_plain(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    m = nnx.Module()
    m.register_variable("0", nnx.Param(1))
    m.update(
        nnx.State({"0": nnx.Param(2)}),
        nnx.State({"new_var": nnx.Param(5)}),
    )
    assert m._variables["new_var"].value == 5
    with pytest.raises(KeyError, match="Unknown child module key"):
        m.update(nnx.State({"sub": nnx.State({"x": 1})}))

    gdef, _ = nnx.split(m)
    m_empty_merge = nnx.merge(gdef)
    assert isinstance(m_empty_merge, nnx.Module)
    m_seq_merge = nnx.merge(gdef, [nnx.State({"new_var": nnx.Param(7)})])
    assert m_seq_merge._variables["new_var"].value == 7
    s_seq_merge = nnx.merge(nnx.State({"a": 1}), [nnx.State({"b": 2})])
    assert s_seq_merge["b"] == 2

    def _make_lin(
        *args: ArrayLike,
        **_kwargs: ArrayLike,
    ) -> nnx.Module:
        """Construct a Linear module from ``args``.

        Args:
            *args: Positional arguments ``(in_features, out_features)``.
            **_kwargs: Unused keyword arguments.

        Returns:
            Constructed ``Linear`` module.
        """
        d_in = int(args[0]) if isinstance(args[0], (int, float)) else 2
        d_out = int(args[1]) if isinstance(args[1], (int, float)) else 2
        return nnx.Linear(d_in, d_out)

    def _double_fn(
        *args: ArrayLike,
        **_kwargs: ArrayLike,
    ) -> ArrayLike:
        """Double the first argument and add the optional second argument.

        Args:
            *args: Positional array/scalar arguments.
            **_kwargs: Unused keyword arguments.

        Returns:
            Scaled ``ArrayLike`` result.
        """
        if len(args) == 0:
            return 0.0
        first = ensure_array(args[0]) * 2.0
        if len(args) > 1:
            return first + ensure_array(args[1])
        return first

    def _tuple_fn(
        *args: ArrayLike,
        **_kwargs: ArrayLike,
    ) -> tuple[JaxArray, JaxArray]:
        """Return a pair ``(x, 2 * x)``.

        Args:
            *args: Positional arguments.
            **_kwargs: Unused keyword arguments.

        Returns:
            Tuple ``(x, 2 * x)``.
        """
        x = ensure_array(args[0])
        return x, x * 2.0

    def _inc_fn(
        *args: ArrayLike,
        **_kwargs: ArrayLike,
    ) -> ArrayLike:
        """Add 1.0 to the first argument (and optional second argument).

        Args:
            *args: Positional array/scalar arguments.
            **_kwargs: Unused keyword arguments.

        Returns:
            Incremented ``ArrayLike`` result.
        """
        if len(args) == 0:
            return 1.0
        first = ensure_array(args[0]) + 1.0
        if len(args) > 1:
            return first + ensure_array(args[1])
        return first

    class _SingleArgCallable(nnx.Module):
        """Callable Module taking a single positional argument ``x``."""

        def __call__(self, x: ArrayLike, **_kwargs: ArrayLike) -> JaxArray:
            """Add 1.0 to ``x``.

            Args:
                x: Input array.
                **_kwargs: Unused keyword arguments.

            Returns:
                Incremented ``JaxArray``.
            """
            return ensure_array(x) + 1.0

    _one_arg_fn = _SingleArgCallable()

    def _ret_mod_fn(*_args: ArrayLike, **_kwargs: ArrayLike) -> nnx.Module:
        """Return an empty ``Module`` instance.

        Args:
            *_args: Unused positional arguments.
            **_kwargs: Unused keyword arguments.

        Returns:
            Empty ``Module`` instance.
        """
        return nnx.Module()

    j_cls_default = nnx.Jit(nnx.Linear)
    assert np.asarray(j_cls_default(np.ones((1, 1), dtype=np.float32))).shape == (1, 1)
    j_cls_opt = nnx.Jit(nnx.RoPE)
    assert np.asarray(j_cls_opt(np.ones((1, 2, 4), dtype=np.float32))).shape == (1, 2, 4)
    j_cls_args = nnx.Jit(nnx.Linear, module_init_args=(2, 3))
    assert np.asarray(j_cls_args(np.ones((1, 2), dtype=np.float32))).shape == (1, 3)

    j_factory = nnx.Jit(
        _make_lin,
        module_init_args=(2, 2),
        module_init_kwargs={"scale": 1},
    )
    assert np.asarray(j_factory(np.ones((1, 2), dtype=np.float32))).shape == (1, 2)
    j_non_mod_init = nnx.Jit(_double_fn, module_init_args=(1,))
    assert float(np.asarray(j_non_mod_init(3.0))) == 6.0
    j_ret_mod = nnx.Jit(_ret_mod_fn)
    assert np.asarray(j_ret_mod(1.0)).shape == (0,)
    j_inactive = nnx.Jit(nnx.Module())
    assert float(np.asarray(j_inactive(5.0))) == 5.0
    assert np.asarray(j_inactive()).shape == (0,)
    r_fn = nnx.Remat(_double_fn)
    assert float(np.asarray(r_fn(3.0))) == 6.0

    seq_inactive = nnx.Sequential(nnx.Module())
    assert seq_inactive(4.0) == 4.0

    gn_2d = nnx.GroupNorm(num_features=4, num_groups=2)
    assert gn_2d(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)

    v_fn = nnx.Vmap(_double_fn)
    assert float(np.asarray(v_fn())) == 0.0
    assert float(np.asarray(v_fn(3.0))) == 6.0
    assert np.asarray(v_fn(np.ones((2, 2), dtype=np.float32), 1.0)).shape == (2, 2)

    v_seq_axes = nnx.Vmap(_double_fn, in_axes=(0, None), out_axes=0)
    assert np.asarray(
        v_seq_axes(
            np.ones((2, 2), dtype=np.float32),
            np.ones((2,), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        )
    ).shape == (2, 2)

    v_none_axes = nnx.Vmap(_double_fn, in_axes=None, axis_size=2)
    assert np.asarray(v_none_axes(np.ones((2,), dtype=np.float32))).shape == (2, 2)

    v_tuple = nnx.Vmap(_tuple_fn, in_axes=0, out_axes=(0, 0))
    vt_res = v_tuple(np.ones((2, 3), dtype=np.float32))
    assert isinstance(vt_res, tuple) and vt_res[0].shape == (2, 3)

    s_single = nnx.Scan(_inc_fn, in_axes=0, out_axes=0, reverse=True)
    assert float(np.asarray(s_single())) == 1.0
    assert np.asarray(s_single(np.ones((3, 2), dtype=np.float32))).shape == (3, 2)

    s_carry_only = nnx.Scan(_one_arg_fn, in_axes=0, out_axes=0, scan_output=False)
    c_final = s_carry_only(np.zeros((2,), dtype=np.float32), np.ones((3, 2), dtype=np.float32))
    assert np.asarray(c_final).shape == (2,)

    s_ret_mod = nnx.Scan(_ret_mod_fn, in_axes=0, out_axes=0)
    res_ret_mod = s_ret_mod(np.zeros((2,)), np.ones((2, 2)))
    assert isinstance(res_ret_mod, tuple) and res_ret_mod[1].shape == (2, 2)

    s_inactive = nnx.Scan(nnx.Module(), in_axes=0, out_axes=0)
    res_inactive = s_inactive(np.zeros((2,)), np.ones((2, 2)))
    assert isinstance(res_inactive, tuple) and res_inactive[1].shape == (2, 2)

    p_fn = nnx.Pmap(_inc_fn)
    assert float(np.asarray(p_fn())) == 1.0
    assert float(np.asarray(p_fn(2.0))) == 3.0
    assert np.asarray(p_fn(np.ones((2, 2), dtype=np.float32), 1.0)).shape == (2, 2)
