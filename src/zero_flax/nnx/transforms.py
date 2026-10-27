"""Functional transformations (`Jit`, `Remat`, `Vmap`, `Scan`, `Pmap`) for zero-flax NNX.

This module implements stateful NNX transformation wrappers that instantiate or wrap
an `nnx.Module` (or callable) and execute it under JIT compilation, gradient
checkpointing (`Remat`), vectorization (`Vmap`), sequential loop unrolling (`Scan`),
or parallel device mapping (`Pmap`).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Optional, Protocol, Tuple, Union

import zero_jax
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from typing_extensions import TypeAlias, TypeGuard
from zero_jax import ad_checkpoint as jad_checkpoint
from zero_jax import lax as jlax
from zero_jax import numpy as jnp
from zero_jax._types import PyTree as JaxPyTree
from zero_jax._types import VariadicCallable
from zero_jax.sharding import Sharding
from zero_jax.stages import Lowered, Traced

from zero_flax.nnx.filterlib import FilterLiteral, to_predicate
from zero_flax.nnx.module import Module, merge, split
from zero_flax.nnx.rnglib import Rngs, RngStream
from zero_flax.nnx.state import State
from zero_flax.types import (
    ArrayLike,
    AxisName,
    InitializerLike,
    JaxArray,
    MetadataValue,
    ensure_array,
    sanitize_dtype_str,
)

InitArgValue: TypeAlias = Union[
    InitializerLike,
    Module,
    Rngs,
    RngStream,
    Sequence[int],
    int,
    float,
    str,
    bool,
    None,
]
InitArgsType: TypeAlias = Optional[Sequence[InitArgValue]]
InitKwargsType: TypeAlias = Optional[Mapping[str, InitArgValue]]
CarryStruct: TypeAlias = Union[JaxArray, Tuple[JaxArray, ...]]
TransformOutput: TypeAlias = Union[JaxArray, Tuple[Union[JaxArray, Tuple[JaxArray, ...]], ...]]


class TransformTargetProtocol(Protocol):
    """Protocol for callables or module factories wrapped by NNX transformations."""

    def __call__(
        self,
        *args: Union[ArrayLike, Tuple[ArrayLike, ...]],
        **kwargs: ArrayLike,
    ) -> Union[
        Module,
        ArrayLike,
        Tuple[Union[ArrayLike, Tuple[ArrayLike, ...]], ...],
    ]:
        """Invoke the target callable or module constructor.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Constructed ``Module``, output ``ArrayLike``, or tuple of ``ArrayLike`` tensors.
        """


class PolicyCallableProtocol(Protocol):
    """Protocol for rematerialization checkpointing policy callables."""

    def __call__(self, *args: ArrayLike) -> bool:
        """Evaluate whether an intermediate value should be checkpointed.

        Args:
            *args: Policy arguments.

        Returns:
            Boolean decision.
        """


class TwoIntModuleClassProtocol(Protocol):
    """Protocol for 2-integer ``Module`` classes such as ``Linear(in_features, out_features)``."""

    def __call__(
        self,
        in_features: int,
        out_features: int,
    ) -> Module:
        """Instantiate the 2-integer module class.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.

        Returns:
            Constructed ``Module`` instance.
        """


class ModuleClassProtocol(Protocol):
    """Protocol for ``Module`` classes taking optional or arbitrary initialization arguments."""

    def __call__(
        self,
        *args: InitArgValue,
        **kwargs: InitArgValue,
    ) -> Module:
        """Instantiate the module class.

        Args:
            *args: Positional initialization arguments.
            **kwargs: Keyword initialization arguments.

        Returns:
            Constructed ``Module`` instance.
        """


class ZeroArgModuleFactoryProtocol(Protocol):
    """Protocol for zero-argument factory callables returning a ``Module``."""

    def __call__(self) -> Module:
        """Construct and return a ``Module`` instance.

        Returns:
            Constructed ``Module`` instance.
        """


ModuleTarget: TypeAlias = Union[
    Module,
    TwoIntModuleClassProtocol,
    ModuleClassProtocol,
    ZeroArgModuleFactoryProtocol,
    TransformTargetProtocol,
]
WrappedCallableType: TypeAlias = Union[
    Module,
    TwoIntModuleClassProtocol,
    ZeroArgModuleFactoryProtocol,
    TransformTargetProtocol,
]


def _is_module_class(
    target: ModuleTarget,
) -> TypeGuard[ModuleClassProtocol]:
    """Check whether ``target`` is a ``Module`` class type.

    Args:
        target: Candidate transformation target.

    Returns:
        ``True`` if ``target`` is a class inheriting from ``Module``.
    """
    return isinstance(target, type)


def _is_zero_arg_factory(
    target: Union[TwoIntModuleClassProtocol, TransformTargetProtocol, ZeroArgModuleFactoryProtocol],
) -> TypeGuard[ZeroArgModuleFactoryProtocol]:
    """Check whether ``target`` is a zero-argument callable factory.

    Args:
        target: Candidate transformation target.

    Returns:
        ``True`` if ``target`` is a zero-argument callable.
    """
    return len(inspect.signature(target).parameters) == 0


def _is_general_transform_callable(
    target: Union[TwoIntModuleClassProtocol, TransformTargetProtocol, ZeroArgModuleFactoryProtocol],
) -> TypeGuard[TransformTargetProtocol]:
    """Narrow ``target`` to ``TransformTargetProtocol``.

    Args:
        target: Candidate transformation target.

    Returns:
        ``True`` if ``target`` is callable.
    """
    return callable(target)


def _count_required_positional_params(
    target: Union[ModuleClassProtocol, TransformTargetProtocol],
) -> Tuple[int, bool]:
    """Return ``(required_positional_count, accepts_two_or_more)`` for ``target``.

    Args:
        target: Callable or ``Module`` class to inspect.

    Returns:
        Pair ``(num_required, accepts_two_args)``.
    """
    sig = inspect.signature(target)
    req = 0
    pos_cap = 0
    has_varargs = False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos_cap += 1
            if param.default is inspect.Parameter.empty:
                req += 1
    return req, (has_varargs or pos_cap >= 2)


def _instantiate_target(
    target: ModuleTarget,
    init_args: InitArgsType,
    init_kwargs: InitKwargsType,
) -> WrappedCallableType:
    """Instantiate ``target`` if it is a class or factory callable, or return it if already an instance.

    Args:
        target: A ``Module`` instance, ``Module`` subclass, or factory callable.
        init_args: Optional positional initialization arguments.
        init_kwargs: Optional keyword initialization arguments.

    Returns:
        Instantiated ``Module`` or callable function.

    Raises:
        TypeError: If ``target`` is a ``Module`` class requiring positional initialization
            arguments that were not provided in ``init_args`` or ``init_kwargs``.
    """
    if isinstance(target, Module):
        return target
    resolved_args: Tuple[InitArgValue, ...] = tuple(init_args) if init_args is not None else ()
    resolved_kwargs: dict[str, InitArgValue] = dict(init_kwargs) if init_kwargs is not None else {}

    if _is_module_class(target):
        if len(resolved_args) == 0 and len(resolved_kwargs) == 0:
            req_count, _ = _count_required_positional_params(target)
            if req_count > 0:
                raise TypeError(
                    f"Module class {target!r} requires {req_count} positional initialization "
                    f"argument(s); pass module_init_args or module_init_kwargs, or pass an "
                    f"already-instantiated Module."
                )
            return target()
        return target(*resolved_args, **resolved_kwargs)

    if _is_zero_arg_factory(target):
        return target()

    if (init_args is not None or init_kwargs is not None) and _is_general_transform_callable(
        target
    ):
        arr_args: Tuple[ArrayLike, ...] = tuple(
            a for a in resolved_args if isinstance(a, (int, float, bool))
        )
        arr_kwargs: dict[str, ArrayLike] = {
            k: v for k, v in resolved_kwargs.items() if isinstance(v, (int, float, bool))
        }
        created = target(*arr_args, **arr_kwargs)
        if isinstance(created, Module):
            return created
    return target


def _slice_along_axis(arr: JaxArray, axis: int, index: int) -> JaxArray:
    """Slice ``arr`` at ``index`` along ``axis``.

    Args:
        arr: Input tensor to slice.
        axis: Dimension index along which to extract element ``index``.
        index: Element position along ``axis``.

    Returns:
        Sliced ``JaxArray`` with ``axis`` removed.
    """
    norm_ax = axis if axis >= 0 else max(0, arr.ndim + axis)
    moved: JaxArray = jnp.swapaxes(arr, 0, norm_ax) if norm_ax != 0 else arr
    res: JaxArray = moved[index]
    return res


def _is_callable_target(
    instance: WrappedCallableType,
) -> TypeGuard[TransformTargetProtocol]:
    """Check whether ``instance`` has a ``__call__`` method.

    Args:
        instance: Wrapped ``Module`` or callable target.

    Returns:
        ``True`` if ``instance`` is callable, else ``False``.
    """
    return callable(instance)


def _coerce_carry_struct(raw_carry: Union[ArrayLike, Tuple[ArrayLike, ...]]) -> CarryStruct:
    """Coerce a single array or tuple of arrays into a ``CarryStruct``.

    Args:
        raw_carry: Single array or tuple of arrays representing a recurrent carry.

    Returns:
        ``JaxArray`` or ``Tuple[JaxArray, ...]`` preserving tuple structure.
    """
    if isinstance(raw_carry, tuple):
        return tuple(ensure_array(item) for item in raw_carry)
    return ensure_array(raw_carry)


def _call_wrapped(
    instance: WrappedCallableType,
    *args: Union[ArrayLike, Tuple[ArrayLike, ...]],
    **kwargs: ArrayLike,
) -> TransformOutput:
    """Invoke a wrapped ``Module`` or callable with ``*args`` and ``**kwargs``.

    Args:
        instance: Target ``Module`` or callable.
        *args: Positional array arguments.
        **kwargs: Keyword array arguments.

    Returns:
        ``JaxArray`` or tuple of ``JaxArray``s produced by ``instance(*args, **kwargs)``.

    Raises:
        TypeError: If ``instance`` is not callable or returns a ``Module`` instead of tensor output.
    """
    if not _is_callable_target(instance):
        raise TypeError(f"Wrapped transformation target {instance!r} is not callable.")
    raw_res = instance(*args, **kwargs)
    if isinstance(raw_res, tuple):
        return tuple(_coerce_carry_struct(item) for item in raw_res)
    if isinstance(raw_res, Module):
        raise TypeError(
            f"Wrapped transformation target {instance!r} returned a Module instead of an ArrayLike."
        )
    return ensure_array(raw_res)


def _stack_step_results(
    step_results: Sequence[TransformOutput],
    out_axes: Union[int, Sequence[Optional[int]], None],
) -> TransformOutput:
    """Stack a non-empty sequence of per-step ``TransformOutput`` results along ``out_axes``.

    Args:
        step_results: Sequence of per-step array or tuple-of-array outputs.
        out_axes: Output axis specification (integer, ``None``, or sequence of optional ints).

    Returns:
        Stacked ``JaxArray`` or tuple of ``JaxArray``s.
    """
    first_res = step_results[0]
    default_ax = out_axes if isinstance(out_axes, int) else 0
    if isinstance(first_res, tuple):
        tuple_steps = [s for s in step_results if isinstance(s, tuple)]
        num_tuple_elems = len(first_res)
        stacked_tuple: list[Union[JaxArray, Tuple[JaxArray, ...]]] = []
        for t_i in range(num_tuple_elems):
            elem_axis = (
                out_axes[t_i]
                if isinstance(out_axes, Sequence) and t_i < len(out_axes)
                else (None if out_axes is None else default_ax)
            )
            first_elem = first_res[t_i]
            if elem_axis is None or isinstance(first_elem, tuple):
                stacked_tuple.append(first_elem)
            else:
                pieces = [
                    jnp.expand_dims(ensure_array(step_r[t_i]), axis=elem_axis)
                    for step_r in tuple_steps
                ]
                stacked_tuple.append(jnp.concatenate(pieces, axis=elem_axis))
        return tuple(stacked_tuple)

    if out_axes is None:
        return first_res
    arr_steps = [s for s in step_results if not isinstance(s, tuple)]
    slices_out = [jnp.expand_dims(step_r, axis=default_ax) for step_r in arr_steps]
    return jnp.concatenate(slices_out, axis=default_ax)


def _resolve_sharding_arg(
    res: Union[Sharding, Sequence[Optional[Sharding]], MetadataValue],
) -> Union[Sharding, Sequence[Optional[Sharding]], None]:
    """Coerce ``in_axis_resources`` / ``out_axis_resources`` into ``zero_jax.jit`` sharding arguments.

    Args:
        res: Sharding specification or metadata resource descriptor.

    Returns:
        ``Sharding``, sequence of ``Optional[Sharding]``, or ``None``.
    """
    if isinstance(res, Sharding):
        return res
    if isinstance(res, (tuple, list)) and all(
        isinstance(item, Sharding) or item is None for item in res
    ):
        return [item if isinstance(item, Sharding) else None for item in res]
    return None


def _is_currently_tracing() -> bool:
    """Return whether ``ml_switcheroo_compiler`` graph tracing is currently active.

    Returns:
        ``True`` if ``_tracer.is_tracing`` is active, else ``False``.
    """
    return bool(_tracer.is_tracing)


def _restore_tracer_state(prev_tracing: bool) -> None:
    """Stop active tracing on error and restore ``_tracer.is_tracing`` to ``prev_tracing``.

    Args:
        prev_tracing: Previous boolean tracing state before ``_jit_fn`` invocation.
    """
    _tracer.stop_tracing()
    _tracer.is_tracing = prev_tracing


def _coerce_transform_output(raw: JaxPyTree) -> TransformOutput:
    """Coerce a ``zero_jax`` PyTree output into a ``TransformOutput``.

    Args:
        raw: Output returned by ``zero_jax.jit``, ``vmap``, ``pmap``, or ``checkpoint``.

    Returns:
        Coerced ``TransformOutput`` (``JaxArray`` or tuple of ``JaxArray``s).
    """
    if isinstance(raw, tuple):
        out_items: list[Union[JaxArray, Tuple[JaxArray, ...]]] = [
            tuple(
                ensure_array(sub)
                for sub in item
                if isinstance(sub, (JaxArray, int, float, bool, list, tuple))
            )
            if isinstance(item, tuple)
            else ensure_array(item)
            for item in raw
            if isinstance(item, (JaxArray, int, float, bool, list, tuple))
        ]
        return tuple(out_items)
    return (
        ensure_array(raw)
        if isinstance(raw, (JaxArray, int, float, bool, list))
        else jnp.asarray(0.0)
    )


class Jit(Module):
    """Just-In-Time (JIT) compilation wrapper for NNX modules and callables.

    Splits ``wrapped_module`` into ``(GraphDef, State)``, passes ``State`` as an explicit
    traced PyTree into ``zero_jax.jit``, and merges the updated ``State`` back into the
    module on return.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        in_axis_resources: Input sharding specification.
        out_axis_resources: Output sharding specification.
        module_init_args: Positional arguments used to initialize ``module_constructor``.
        module_init_kwargs: Keyword arguments used to initialize ``module_constructor``.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    in_axis_resources: Union[Sharding, Sequence[Optional[Sharding]], MetadataValue]
    out_axis_resources: Union[Sharding, Sequence[Optional[Sharding]], MetadataValue]
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType
    _jit_fn: VariadicCallable[JaxPyTree]
    _trace_count: int

    def __init__(
        self,
        module_constructor: ModuleTarget,
        in_axis_resources: Union[Sharding, Sequence[Optional[Sharding]], MetadataValue] = None,
        out_axis_resources: Union[Sharding, Sequence[Optional[Sharding]], MetadataValue] = None,
        module_init_args: InitArgsType = None,
        module_init_kwargs: InitKwargsType = None,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a Jit transformation wrapper.

        Args:
            module_constructor: ``Module`` instance, class, or factory callable to JIT-compile.
            in_axis_resources: Optional input sharding axis resources.
            out_axis_resources: Optional output sharding axis resources.
            module_init_args: Optional positional arguments passed to ``module_constructor``.
            module_init_kwargs: Optional keyword arguments passed to ``module_constructor``.
            **kwargs: Additional static configuration metadata.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
        self.in_axis_resources = in_axis_resources
        self.out_axis_resources = out_axis_resources
        self.module_init_args = module_init_args
        self.module_init_kwargs = module_init_kwargs
        self._trace_count = 0
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst

        in_sh = _resolve_sharding_arg(in_axis_resources)
        out_sh = _resolve_sharding_arg(out_axis_resources)
        self._jit_fn = zero_jax.jit(
            self._jit_entry,
            in_shardings=in_sh,
            out_shardings=out_sh,
        )
        self.finish_init()

    def _jit_entry(self, *args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """Pure entry function compiled by ``zero_jax.jit``.

        When ``self.wrapped_module`` is an ``nnx.Module``, ``args[0]`` is the split
        ``State`` PyTree; reconstructs the module via ``merge``, runs the forward pass,
        and returns ``(out, updated_state)``.

        Args:
            *args: ``(state, *call_args)`` for modules, or ``*call_args`` for callables.
            **kwargs: Keyword array arguments.

        Returns:
            ``(out, new_state)`` for modules, or ``out`` for callables.
        """
        self._trace_count += 1
        kw_arrs: dict[str, ArrayLike] = {
            k: ensure_array(v)
            for k, v in kwargs.items()
            if isinstance(v, (JaxArray, int, float, bool, list, tuple))
        }
        if (
            isinstance(self.wrapped_module, Module)
            and len(args) >= 1
            and isinstance(args[0], State)
        ):
            state_arg = args[0]
            gdef, _ = split(self.wrapped_module)
            reconstructed = merge(gdef, state_arg)
            pos_arrs: list[Union[ArrayLike, Tuple[ArrayLike, ...]]] = [
                ensure_array(a)
                for a in args[1:]
                if isinstance(a, (JaxArray, int, float, bool, list, tuple))
            ]
            out = _call_wrapped(reconstructed, *pos_arrs, **kw_arrs)
            _, new_state = split(reconstructed)
            return out, new_state

        pos_only: list[Union[ArrayLike, Tuple[ArrayLike, ...]]] = [
            ensure_array(a)
            for a in args
            if isinstance(a, (JaxArray, int, float, bool, list, tuple))
        ]
        return _call_wrapped(self.wrapped_module, *pos_only, **kw_arrs)

    @property
    def trace_count(self) -> int:
        """Return the number of distinct JIT compilations performed by this wrapper.

        Returns:
            Integer trace count.
        """
        return self._trace_count

    def trace(self, *args: ArrayLike, **kwargs: ArrayLike) -> Traced:
        """Return a :class:`zero_jax.stages.Traced` stage for sample arguments.

        Args:
            *args: Sample positional array arguments.
            **kwargs: Sample keyword array arguments.

        Returns:
            ``zero_jax.stages.Traced`` object.
        """
        _ = (args, kwargs)
        return Traced(fun=self._jit_fn)

    def lower(self, *args: ArrayLike, **kwargs: ArrayLike) -> Lowered:
        """Return a :class:`zero_jax.stages.Lowered` stage for sample arguments.

        Args:
            *args: Sample positional array arguments.
            **kwargs: Sample keyword array arguments.

        Returns:
            ``zero_jax.stages.Lowered`` object.
        """
        _ = (args, kwargs)
        return Lowered(fun=self._jit_fn)

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Execute the JIT-wrapped module or callable on ``*args`` and ``**kwargs``.

        Splits ``self.wrapped_module`` into ``(GraphDef, State)``, executes through
        ``zero_jax.jit`` when not already inside an outer trace, and merges the updated
        ``State`` back into ``self.wrapped_module``.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Output ``JaxArray`` or tuple of ``JaxArray``s of the wrapped module computation.
        """
        if not _is_callable_target(self.wrapped_module):
            return _call_wrapped(self.wrapped_module, *args, **kwargs)

        arr_args = tuple(ensure_array(a) for a in args)
        arr_kwargs = {k: ensure_array(v) for k, v in kwargs.items()}

        if _is_currently_tracing():
            if isinstance(self.wrapped_module, Module):
                gdef, st = split(self.wrapped_module)
                reconstructed = merge(gdef, st)
                out_traced = _call_wrapped(reconstructed, *arr_args, **arr_kwargs)
                _, new_st = split(reconstructed)
                self.wrapped_module.update(new_st)
                return out_traced
            return _call_wrapped(self.wrapped_module, *arr_args, **arr_kwargs)

        if isinstance(self.wrapped_module, Module):
            gdef, st = split(self.wrapped_module)
            prev_graph = _tracer.active_graph
            prev_tracing = _is_currently_tracing()
            try:
                jit_out, jit_new_st = self._jit_fn(st, *arr_args, **arr_kwargs)
                self.wrapped_module.update(jit_new_st)
                return _coerce_transform_output(jit_out)
            except (
                IndexError,
                KeyError,
                RuntimeError,
                ValueError,
                ArithmeticError,
                AttributeError,
                TypeError,
            ):
                _restore_tracer_state(prev_tracing)
                _tracer.active_graph = prev_graph
                reconstructed = merge(gdef, st)
                out_eager = _call_wrapped(reconstructed, *arr_args, **arr_kwargs)
                _, new_st = split(reconstructed)
                self.wrapped_module.update(new_st)
                return out_eager

        prev_graph = _tracer.active_graph
        prev_tracing = _is_currently_tracing()
        try:
            fn_res = self._jit_fn(*arr_args, **arr_kwargs)
            return _coerce_transform_output(fn_res)
        except (
            IndexError,
            KeyError,
            RuntimeError,
            ValueError,
            ArithmeticError,
            AttributeError,
            TypeError,
        ):
            _restore_tracer_state(prev_tracing)
            _tracer.active_graph = prev_graph
            return _call_wrapped(self.wrapped_module, *arr_args, **arr_kwargs)


class Remat(Module):
    """Gradient checkpointing (`Remat`) wrapper for NNX modules and callables.

    Delegates activation checkpointing to ``zero_jax.ad_checkpoint.checkpoint`` (``remat``),
    forwarding ``prevent_cse``, ``static_argnums``, and ``policy``.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        prevent_cse: Whether common subexpression elimination is prevented across checkpoint boundaries.
        policy: Optional checkpointing policy callable or descriptor.
        static_argnums: Positional argument index or indices treated as static.
        module_init_args: Positional arguments used to initialize ``module_constructor``.
        module_init_kwargs: Keyword arguments used to initialize ``module_constructor``.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    prevent_cse: bool
    policy: Optional[Union[PolicyCallableProtocol, str]]
    static_argnums: Union[int, Sequence[int]]
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType

    def __init__(
        self,
        module_constructor: ModuleTarget,
        prevent_cse: bool = True,
        policy: Optional[Union[PolicyCallableProtocol, str]] = None,
        static_argnums: Union[int, Sequence[int]] = (),
        module_init_args: InitArgsType = None,
        module_init_kwargs: InitKwargsType = None,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a Remat gradient-checkpointing wrapper.

        Args:
            module_constructor: ``Module`` instance, class, or factory callable to wrap.
            prevent_cse: Whether to prevent CSE optimizations across rematerialization boundaries.
            policy: Optional checkpointing policy.
            static_argnums: Optional positional argument index or sequence of indices treated as static.
            module_init_args: Optional positional initialization arguments.
            module_init_kwargs: Optional keyword initialization arguments.
            **kwargs: Additional static configuration metadata.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
        self.prevent_cse = bool(prevent_cse)
        self.policy = policy
        self.static_argnums = (
            int(static_argnums)
            if isinstance(static_argnums, int)
            else tuple(int(i) for i in static_argnums)
        )
        self.module_init_args = module_init_args
        self.module_init_kwargs = module_init_kwargs
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst
        self.finish_init()

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Execute the checkpointed module or callable via ``zero_jax.ad_checkpoint.checkpoint``.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Output ``JaxArray`` or tuple of ``JaxArray``s of the wrapped module computation.
        """
        if not _is_callable_target(self.wrapped_module):
            return _call_wrapped(self.wrapped_module, *args, **kwargs)

        def _remat_body(*b_args: JaxPyTree, **b_kwargs: JaxPyTree) -> JaxPyTree:
            """Inner rematerialized callable executed inside ``ad_checkpoint.checkpoint``.

            Args:
                *b_args: Positional array arguments.
                **b_kwargs: Keyword array arguments.

            Returns:
                Checkpointed output tagged with ``checkpoint_name``.
            """
            pos_list: list[Union[ArrayLike, Tuple[ArrayLike, ...]]] = [
                ensure_array(a)
                for a in b_args
                if isinstance(a, (JaxArray, int, float, bool, list, tuple))
            ]
            kw_map: dict[str, ArrayLike] = {
                k: ensure_array(v)
                for k, v in b_kwargs.items()
                if isinstance(v, (JaxArray, int, float, bool, list, tuple))
            }
            raw_out = _call_wrapped(self.wrapped_module, *pos_list, **kw_map)
            return jad_checkpoint.checkpoint_name(raw_out, "remat_output")

        ckpt_fn = jad_checkpoint.checkpoint(
            _remat_body,
            prevent_cse=self.prevent_cse,
            policy=self.policy
            if isinstance(self.policy, str) or self.policy is None
            else self.policy,
            static_argnums=self.static_argnums,
        )
        arr_args = tuple(ensure_array(a) for a in args)
        arr_kwargs = {k: ensure_array(v) for k, v in kwargs.items()}
        res = ckpt_fn(*arr_args, **arr_kwargs)
        return _coerce_transform_output(res)


class Vmap(Module):
    """Vectorizing map (`Vmap`) wrapper mapping an NNX module across a batch axis.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        in_axes: Input batch axis specification.
        out_axes: Output batch axis specification.
        axis_name: Optional logical batch axis name.
        axis_size: Optional explicit batch axis size.
        spmd_axis_name: Optional SPMD axis name.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    in_axes: Union[int, Sequence[Optional[int]], None]
    out_axes: Union[int, Sequence[Optional[int]], None]
    axis_name: Optional[AxisName]
    axis_size: Optional[int]
    spmd_axis_name: Optional[AxisName]
    variable_axes: Optional[Mapping[FilterLiteral, int]]
    split_rngs: FilterLiteral
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType

    def __init__(
        self,
        module_constructor: ModuleTarget,
        in_axes: Union[int, Sequence[Optional[int]], None] = 0,
        out_axes: Union[int, Sequence[Optional[int]], None] = 0,
        axis_name: Optional[AxisName] = None,
        axis_size: Optional[int] = None,
        spmd_axis_name: Optional[AxisName] = None,
        variable_axes: Optional[Mapping[FilterLiteral, int]] = None,
        split_rngs: FilterLiteral = None,
        module_init_args: InitArgsType = None,
        module_init_kwargs: InitKwargsType = None,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a Vmap vectorization wrapper.

        Args:
            module_constructor: ``Module`` instance, class, or factory callable to vectorize.
            in_axes: Integer, ``None``, or sequence specifying which input axis to map over. Defaults to ``0``.
            out_axes: Integer, ``None``, or sequence specifying where the mapped axis appears in the output.
            axis_name: Optional name for the mapped batch axis.
            axis_size: Optional explicit size of the mapped axis.
            spmd_axis_name: Optional SPMD mesh axis name.
            variable_axes: Optional mapping of variable filters to mapped axes.
            split_rngs: Filter specifying which RNG streams to split across batch elements.
            module_init_args: Optional positional initialization arguments.
            module_init_kwargs: Optional keyword initialization arguments.
            **kwargs: Additional static configuration metadata.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.axis_name = axis_name
        self.axis_size = axis_size
        self.spmd_axis_name = spmd_axis_name
        self.variable_axes = variable_axes
        self.split_rngs = split_rngs
        self.module_init_args = module_init_args
        self.module_init_kwargs = module_init_kwargs
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst
        self.finish_init()

    def _resolve_arg_axis(self, arg_index: int) -> Optional[int]:
        """Resolve the mapped input axis for argument at ``arg_index``.

        Args:
            arg_index: Positional argument index.

        Returns:
            Axis index integer or ``None`` if unmapped.
        """
        if self.in_axes is None:
            return None
        if isinstance(self.in_axes, int):
            return self.in_axes
        if arg_index < len(self.in_axes):
            return self.in_axes[arg_index]
        return 0

    def _build_state_axes(self, state: State, prefix: Tuple[str, ...] = ()) -> Optional[State]:
        """Build a parallel ``State`` of integer/``None`` axis specifications from ``self.variable_axes``.

        Args:
            state: Module ``State`` to match.
            prefix: Key path prefix for filter predicate evaluation.

        Returns:
            ``State`` of axis specifications if ``self.variable_axes`` is set, else ``None``.
        """
        if not self.variable_axes:
            return None
        pred_axes = [(to_predicate(flt), int(ax)) for flt, ax in self.variable_axes.items()]
        axes_state = State()
        for key, entry in state.items():
            cur_path = prefix + (key,)
            if isinstance(entry, State):
                sub_axes = self._build_state_axes(entry, cur_path)
                if sub_axes is not None:
                    axes_state[key] = sub_axes
            else:
                matched_ax: Optional[int] = None
                for pred, ax_val in pred_axes:
                    if pred(cur_path, entry):
                        matched_ax = ax_val
                        break
                axes_state[key] = matched_ax
        return axes_state

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Vectorize execution of ``self.wrapped_module`` across ``in_axes`` using ``zero_jax.vmap``.

        Args:
            *args: Positional batched input arrays.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Stacked ``JaxArray`` (or tuple of stacked ``JaxArray``s) of vectorized outputs along ``out_axes``.

        Raises:
            ValueError: If mapped input arguments have inconsistent batch dimension sizes.
        """
        if len(args) == 0 and self.axis_size is None:
            return _call_wrapped(self.wrapped_module, **kwargs)

        arr_args = [ensure_array(a) for a in args]
        arg_axes = [self._resolve_arg_axis(i) for i in range(len(arr_args))]

        batch_len: Optional[int] = self.axis_size
        for arr_a, ax_a in zip(arr_args, arg_axes):
            if ax_a is not None and arr_a.ndim > 0:
                norm_ax = ax_a if ax_a >= 0 else arr_a.ndim + ax_a
                dim_sz = arr_a.shape[norm_ax]
                if batch_len is None:
                    batch_len = dim_sz
                elif dim_sz != batch_len:
                    raise ValueError(
                        f"Vmap got inconsistent batch axis sizes: expected {batch_len}, got {dim_sz}."
                    )
        if batch_len is None:
            return _call_wrapped(self.wrapped_module, *args, **kwargs)

        state_axes_tree: Optional[State] = None
        if isinstance(self.wrapped_module, Module):
            _, cur_st = split(self.wrapped_module)
            state_axes_tree = self._build_state_axes(cur_st)

        ax_name_str = str(self.axis_name) if self.axis_name is not None else "_nnx_vmap_axis"
        spmd_str = str(self.spmd_axis_name) if self.spmd_axis_name is not None else None
        out_ax_spec = self.out_axes if isinstance(self.out_axes, int) else 0
        if (
            len(kwargs) == 0
            and batch_len > 0
            and isinstance(self.out_axes, int)
            and state_axes_tree is None
            and all(a.ndim > 0 for a in arr_args)
            and _is_callable_target(self.wrapped_module)
        ):

            def _vmap_single(*v_args: JaxPyTree) -> JaxPyTree:
                """Single-element callable mapped by ``zero_jax.vmap``.

                Args:
                    *v_args: Sliced positional arguments for one batch element.

                Returns:
                    Output for the single batch element.
                """
                jlax.control_flow.apply_axis_collective("all_gather", jnp.asarray(0.0), ax_name_str)
                s_args: list[Union[ArrayLike, Tuple[ArrayLike, ...]]] = [
                    ensure_array(a)
                    for a in v_args
                    if isinstance(a, (JaxArray, int, float, bool, list, tuple))
                ]
                return _call_wrapped(self.wrapped_module, *s_args)

            vmapped_fn = zero_jax.vmap(
                _vmap_single,
                in_axes=tuple(arg_axes),
                axis_name=ax_name_str,
                axis_size=batch_len,
                out_axes=out_ax_spec,
                spmd_axis_name=spmd_str,
            )
            vmap_out = vmapped_fn(*arr_args)
            return _coerce_transform_output(vmap_out)

        step_results: list[TransformOutput] = []
        for idx in range(batch_len):
            step_args: list[ArrayLike] = []
            for arr_a, ax_a in zip(arr_args, arg_axes):
                if ax_a is not None and arr_a.ndim > 0:
                    step_args.append(_slice_along_axis(arr_a, ax_a, idx))
                else:
                    step_args.append(arr_a)
            if isinstance(self.wrapped_module, Module) and state_axes_tree is not None:
                gdef_v, st_v = split(self.wrapped_module)
                mod_v = merge(gdef_v, st_v)
                step_results.append(_call_wrapped(mod_v, *step_args, **kwargs))
                _, new_st_v = split(mod_v)
                self.wrapped_module.update(new_st_v)
            else:
                step_results.append(_call_wrapped(self.wrapped_module, *step_args, **kwargs))

        return _stack_step_results(step_results, self.out_axes)


class Scan(Module):
    """Stateful sequential scan (`Scan`) wrapper unrolling a module across a sequence axis.

    Delegates sequential loop execution to ``zero_jax.lax.scan`` while tracking and updating
    ``nnx.Module`` state across iterations.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        in_axes: Input sequence axis index.
        out_axes: Output sequence axis index.
        length: Optional explicit sequence length.
        reverse: Whether to scan in reverse order.
        unroll: Loop unroll factor.
        carry_argnum: Index of the carry argument.
        scan_output: Whether the wrapped callable returns ``(carry, y)`` or only ``carry``.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    in_axes: int
    out_axes: int
    length: Optional[int]
    reverse: bool
    unroll: int
    data_transform: Optional[TransformTargetProtocol]
    variable_axes: Optional[Mapping[FilterLiteral, int]]
    variable_broadcast: FilterLiteral
    variable_carry: FilterLiteral
    split_rngs: FilterLiteral
    carry_argnum: int
    scan_output: bool
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType

    def __init__(
        self,
        module_constructor: ModuleTarget,
        in_axes: int = 1,
        out_axes: int = 1,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
        data_transform: Optional[TransformTargetProtocol] = None,
        variable_axes: Optional[Mapping[FilterLiteral, int]] = None,
        variable_broadcast: FilterLiteral = None,
        variable_carry: FilterLiteral = None,
        split_rngs: FilterLiteral = None,
        carry_argnum: int = 0,
        scan_output: bool = True,
        module_init_args: InitArgsType = None,
        module_init_kwargs: InitKwargsType = None,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a Scan loop transformation wrapper.

        Args:
            module_constructor: ``Module`` instance, class, or factory callable to scan.
            in_axes: Input axis along which sequence elements are sliced. Defaults to ``1``.
            out_axes: Output axis along which step outputs are stacked. Defaults to ``1``.
            length: Optional explicit number of scan iterations.
            reverse: Whether to iterate from the end of the sequence to the start.
            unroll: Loop unroll factor.
            data_transform: Optional data transformation hook.
            variable_axes: Optional mapping of variable filters to scanned axes.
            variable_broadcast: Filter for broadcast variables.
            variable_carry: Filter for carried variables.
            split_rngs: Filter for split RNG streams.
            carry_argnum: Positional argument index of the loop carry.
            scan_output: Whether the scanned callable emits per-step outputs alongside carry.
            module_init_args: Optional positional initialization arguments.
            module_init_kwargs: Optional keyword initialization arguments.
            **kwargs: Additional static configuration metadata.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
        self.in_axes = int(in_axes)
        self.out_axes = int(out_axes)
        self.length = length
        self.reverse = bool(reverse)
        self.unroll = int(unroll)
        self.data_transform = data_transform
        self.variable_axes = variable_axes
        self.variable_broadcast = variable_broadcast
        self.variable_carry = variable_carry
        self.split_rngs = split_rngs
        self.carry_argnum = int(carry_argnum)
        self.scan_output = bool(scan_output)
        self.module_init_args = module_init_args
        self.module_init_kwargs = module_init_kwargs
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst
        self.finish_init()

    def __call__(
        self,
        *args: Union[ArrayLike, Tuple[ArrayLike, ...]],
        **kwargs: ArrayLike,
    ) -> TransformOutput:
        """Execute the scanned loop over ``*args`` via ``zero_jax.lax.scan``.

        Args:
            *args: Either ``(carry, *xs)`` or ``(xs,)`` depending on the wrapped module signature.
            **kwargs: Keyword arguments forwarded to each scan step.

        Returns:
            ``(final_carry, stacked_outputs)`` when ``carry`` and ``xs`` are provided,
            or ``stacked_outputs`` when a single tensor is scanned.
        """
        if len(args) == 0:
            return _call_wrapped(self.wrapped_module, **kwargs)
        if len(args) == 1:
            xs = ensure_array(args[0])
            if self.data_transform is not None:
                dt_res = self.data_transform(xs)
                if not isinstance(dt_res, (Module, tuple)):
                    xs = ensure_array(dt_res)
            norm_in_ax = self.in_axes if self.in_axes >= 0 else max(0, xs.ndim + self.in_axes)
            ax = norm_in_ax if norm_in_ax < xs.ndim else 0
            seq_len = self.length if self.length is not None else xs.shape[ax]
            if seq_len == 0:
                return xs
            xs_tm: JaxArray = jnp.moveaxis(xs, ax, 0) if ax != 0 else xs

            def _single_scan_step(
                dummy_c: JaxArray,
                step_x: JaxArray,
            ) -> Tuple[JaxArray, JaxArray]:
                """Single-input scan step executed inside ``zero_jax.lax.scan``.

                Args:
                    dummy_c: Placeholder carry tensor.
                    step_x: Input slice for the current sequence step.

                Returns:
                    Pair ``(dummy_c, step_y)``.
                """
                step_res = _call_wrapped(self.wrapped_module, step_x, **kwargs)
                step_first = step_res[0] if isinstance(step_res, tuple) else step_res
                step_y = step_first[0] if isinstance(step_first, tuple) else step_first
                return dummy_c, step_y

            _, single_stacked = jlax.scan(
                _single_scan_step,
                jnp.zeros((1,), dtype="float32"),
                xs_tm,
                length=seq_len,
                reverse=self.reverse,
                unroll=max(1, self.unroll),
            )
            stacked_single: JaxArray = ensure_array(single_stacked)
            if self.reverse:
                stacked_single = jnp.flip(stacked_single, axis=0)
            return jnp.moveaxis(stacked_single, 0, ax) if ax != 0 else stacked_single

        carry_idx = self.carry_argnum if 0 <= self.carry_argnum < len(args) else 0
        carry: CarryStruct = _coerce_carry_struct(args[carry_idx])
        scanned_indices = [i for i in range(len(args)) if i != carry_idx]
        scanned_arrays: list[JaxArray] = []
        for s_i in scanned_indices:
            arr_s = ensure_array(args[s_i])
            if self.data_transform is not None:
                dt_s = self.data_transform(arr_s)
                if not isinstance(dt_s, (Module, tuple)):
                    arr_s = ensure_array(dt_s)
            scanned_arrays.append(arr_s)

        first_xs = scanned_arrays[0]
        norm_in_ax = self.in_axes if self.in_axes >= 0 else max(0, first_xs.ndim + self.in_axes)
        ax = norm_in_ax if norm_in_ax < first_xs.ndim else 0
        norm_out_ax = self.out_axes if self.out_axes >= 0 else max(0, first_xs.ndim + self.out_axes)
        out_ax = norm_out_ax if norm_out_ax <= first_xs.ndim else 0
        seq_len = self.length if self.length is not None else first_xs.shape[ax]
        if seq_len == 0:
            ref_c = carry[0] if isinstance(carry, tuple) else carry
            empty_shape = list(ref_c.shape)
            empty_shape.insert(out_ax, 0)
            c_dt = sanitize_dtype_str(str(ref_c.dtype).lower())
            empty_out: JaxArray = jnp.zeros(tuple(empty_shape), dtype=c_dt)
            if self.scan_output:
                return carry, empty_out
            return carry

        accepts_two = False
        if _is_callable_target(self.wrapped_module):
            _, accepts_two = _count_required_positional_params(self.wrapped_module)

        scanned_tm: Tuple[JaxArray, ...] = tuple(
            jnp.moveaxis(arr_s, ax, 0) if ax != 0 else arr_s for arr_s in scanned_arrays
        )

        def _carry_scan_step(
            cur_carry: CarryStruct,
            sliced_xs: Tuple[JaxArray, ...],
        ) -> Tuple[CarryStruct, JaxArray]:
            """Carry-and-input scan step executed inside ``zero_jax.lax.scan``.

            Args:
                cur_carry: Current loop carry state.
                sliced_xs: Tuple of input slices for the current sequence step.

            Returns:
                Updated ``(next_carry, step_y)`` pair.
            """
            step_x = sliced_xs[0]
            if _is_callable_target(self.wrapped_module):
                if accepts_two:
                    call_args: list[Union[ArrayLike, Tuple[ArrayLike, ...]]] = list(sliced_xs)
                    call_args.insert(carry_idx, cur_carry)
                    raw_out = self.wrapped_module(*call_args, **kwargs)
                else:
                    raw_out = self.wrapped_module(step_x, **kwargs)
            else:
                raw_out = step_x
            if isinstance(raw_out, tuple) and len(raw_out) == 2:
                next_carry = _coerce_carry_struct(raw_out[0])
                step_y = ensure_array(raw_out[1])
            elif not isinstance(raw_out, (Module, tuple)):
                next_carry = ensure_array(raw_out)
                step_y = next_carry
            else:
                next_carry = cur_carry
                step_y = cur_carry[0] if isinstance(cur_carry, tuple) else cur_carry
            return next_carry, step_y

        scan_res = jlax.scan(
            _carry_scan_step,
            carry,
            scanned_tm,
            length=seq_len,
            reverse=self.reverse,
            unroll=max(1, self.unroll),
        )
        final_carry: CarryStruct = _coerce_carry_struct(scan_res[0])
        stacked_0: JaxArray = ensure_array(scan_res[1])
        if self.reverse:
            stacked_0 = jnp.flip(stacked_0, axis=0)
        stacked: JaxArray = jnp.moveaxis(stacked_0, 0, out_ax) if out_ax != 0 else stacked_0
        if self.scan_output:
            return final_carry, stacked
        return final_carry


class Pmap(Module):
    """Parallel map (`Pmap`) wrapper executing an NNX module across parallel devices.

    Delegates single-program multi-device execution to ``zero_jax.pmap`` and validates
    device availability via ``zero_chex.assert_devices_available``.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        axis_name: Optional parallel axis name.
        in_axes: Input device axis index.
        out_axes: Output device axis index.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    axis_name: Optional[AxisName]
    in_axes: int
    out_axes: int
    static_broadcasted_argnums: Sequence[int]
    devices: Optional[Sequence[str]]
    backend: Optional[str]
    axis_size: Optional[int]
    donate_argnums: Sequence[int]
    global_arg_shapes: Optional[Sequence[Tuple[int, ...]]]
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType

    def __init__(
        self,
        module_constructor: ModuleTarget,
        axis_name: Optional[AxisName] = None,
        *,
        in_axes: int = 0,
        out_axes: int = 0,
        static_broadcasted_argnums: Sequence[int] = (),
        devices: Optional[Sequence[str]] = None,
        backend: Optional[str] = None,
        axis_size: Optional[int] = None,
        donate_argnums: Sequence[int] = (),
        global_arg_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        module_init_args: InitArgsType = None,
        module_init_kwargs: InitKwargsType = None,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a Pmap parallel device mapping wrapper.

        Args:
            module_constructor: ``Module`` instance, class, or factory callable to map across devices.
            axis_name: Optional name for the parallel device axis.
            in_axes: Input axis index mapped across devices. Defaults to ``0``.
            out_axes: Output axis index mapped across devices. Defaults to ``0``.
            static_broadcasted_argnums: Indices of static broadcasted arguments.
            devices: Optional sequence of target device identifiers.
            backend: Optional execution backend name (`"cpu"`, `"gpu"`, `"tpu"`).
            axis_size: Optional explicit size of the parallel axis.
            donate_argnums: Argument indices whose buffers can be donated.
            global_arg_shapes: Optional global shapes for multi-host execution.
            module_init_args: Optional positional initialization arguments.
            module_init_kwargs: Optional keyword initialization arguments.
            **kwargs: Additional static configuration metadata.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
        self.axis_name = axis_name
        self.in_axes = int(in_axes)
        self.out_axes = int(out_axes)
        self.static_broadcasted_argnums = tuple(int(i) for i in static_broadcasted_argnums)
        self.devices = devices
        self.backend = backend
        self.axis_size = axis_size
        self.donate_argnums = tuple(int(i) for i in donate_argnums)
        self.global_arg_shapes = global_arg_shapes
        self.module_init_args = module_init_args
        self.module_init_kwargs = module_init_kwargs
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst
        self.finish_init()

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Execute ``self.wrapped_module`` across the parallel device axis via ``zero_jax.pmap``.

        Args:
            *args: Positional device-sharded input arrays.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Device-mapped output ``JaxArray`` or tuple of ``JaxArray``s.

        Raises:
            ValueError: If non-broadcasted input arguments have inconsistent device axis sizes.
        """
        from zero_chex import assert_devices_available

        assert_devices_available(1, devtype="cpu")

        if len(args) == 0 and self.axis_size is None:
            return _call_wrapped(self.wrapped_module, **kwargs)

        arr_args = [ensure_array(a) for a in args]
        num_devices: Optional[int] = self.axis_size
        for arg_idx, arr_a in enumerate(arr_args):
            if arg_idx not in self.static_broadcasted_argnums and arr_a.ndim > 0:
                norm_ax = self.in_axes if self.in_axes >= 0 else arr_a.ndim + self.in_axes
                dim_sz = arr_a.shape[norm_ax]
                if num_devices is None:
                    num_devices = dim_sz
                elif dim_sz != num_devices:
                    raise ValueError(
                        f"Pmap got inconsistent device axis sizes: expected {num_devices}, got {dim_sz}."
                    )
        if num_devices is None:
            return _call_wrapped(self.wrapped_module, *args, **kwargs)

        ax_name_str = str(self.axis_name) if self.axis_name is not None else "_nnx_pmap_axis"
        if (
            len(kwargs) == 0
            and num_devices > 0
            and len(self.static_broadcasted_argnums) == 0
            and all(a.ndim > 0 for a in arr_args)
            and _is_callable_target(self.wrapped_module)
        ):

            def _pmap_single(*p_args: JaxPyTree) -> JaxPyTree:
                """Single-replica callable mapped by ``zero_jax.pmap``.

                Args:
                    *p_args: Sliced positional arguments for one device replica.

                Returns:
                    Output for the single device replica.
                """
                jlax.control_flow.apply_axis_collective("all_gather", jnp.asarray(0.0), ax_name_str)
                s_args: list[Union[ArrayLike, Tuple[ArrayLike, ...]]] = [
                    ensure_array(a)
                    for a in p_args
                    if isinstance(a, (JaxArray, int, float, bool, list, tuple))
                ]
                return _call_wrapped(self.wrapped_module, *s_args)

            pmapped_fn = zero_jax.pmap(
                _pmap_single,
                axis_name=ax_name_str,
                in_axes=self.in_axes,
                out_axes=self.out_axes,
                devices=self.devices,
                axis_size=num_devices,
                backend=self.backend,
                donate_argnums=self.donate_argnums,
            )
            pmap_out = pmapped_fn(*arr_args)
            return _coerce_transform_output(pmap_out)

        step_results: list[TransformOutput] = []
        for idx in range(num_devices):
            step_args: list[ArrayLike] = []
            for arg_idx, arr_a in enumerate(arr_args):
                if arg_idx not in self.static_broadcasted_argnums and arr_a.ndim > 0:
                    step_args.append(_slice_along_axis(arr_a, self.in_axes, idx))
                else:
                    step_args.append(arr_a)
            step_results.append(_call_wrapped(self.wrapped_module, *step_args, **kwargs))

        return _stack_step_results(step_results, self.out_axes)


__all__ = [
    "InitArgValue",
    "InitArgsType",
    "InitKwargsType",
    "Jit",
    "ModuleTarget",
    "Pmap",
    "PolicyCallableProtocol",
    "Remat",
    "Scan",
    "TransformOutput",
    "TransformTargetProtocol",
    "Vmap",
]
