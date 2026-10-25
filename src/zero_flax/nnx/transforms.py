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

from typing_extensions import TypeAlias, TypeGuard
from zero_jax import numpy as jnp

from zero_flax.nnx.filterlib import FilterLiteral
from zero_flax.nnx.module import Module
from zero_flax.nnx.rnglib import Rngs, RngStream
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
TransformOutput: TypeAlias = Union[JaxArray, Tuple[JaxArray, ...]]


class TransformTargetProtocol(Protocol):
    """Protocol for callables or module factories wrapped by NNX transformations."""

    def __call__(
        self,
        *args: ArrayLike,
        **kwargs: ArrayLike,
    ) -> Union[Module, ArrayLike, Tuple[ArrayLike, ...]]:
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


def _call_wrapped(
    instance: WrappedCallableType,
    *args: ArrayLike,
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
        return tuple(ensure_array(item) for item in raw_res)
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
        stacked_tuple: list[JaxArray] = []
        for t_i in range(num_tuple_elems):
            elem_axis = (
                out_axes[t_i]
                if isinstance(out_axes, Sequence) and t_i < len(out_axes)
                else (None if out_axes is None else default_ax)
            )
            if elem_axis is None:
                stacked_tuple.append(first_res[t_i])
            else:
                pieces = [jnp.expand_dims(step_r[t_i], axis=elem_axis) for step_r in tuple_steps]
                stacked_tuple.append(jnp.concatenate(pieces, axis=elem_axis))
        return tuple(stacked_tuple)

    if out_axes is None:
        return first_res
    arr_steps = [s for s in step_results if not isinstance(s, tuple)]
    slices_out = [jnp.expand_dims(step_r, axis=default_ax) for step_r in arr_steps]
    return jnp.concatenate(slices_out, axis=default_ax)


class Jit(Module):
    """Just-In-Time (JIT) compilation wrapper for NNX modules and callables.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        in_axis_resources: Input sharding specification.
        out_axis_resources: Output sharding specification.
        module_init_args: Positional arguments used to initialize ``module_constructor``.
        module_init_kwargs: Keyword arguments used to initialize ``module_constructor``.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    in_axis_resources: MetadataValue
    out_axis_resources: MetadataValue
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType

    def __init__(
        self,
        module_constructor: ModuleTarget,
        in_axis_resources: MetadataValue = None,
        out_axis_resources: MetadataValue = None,
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
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst
        self.finish_init()

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Execute the JIT-wrapped module or callable on ``*args`` and ``**kwargs``.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Output ``JaxArray`` or tuple of ``JaxArray``s of the wrapped module computation.
        """
        return _call_wrapped(self.wrapped_module, *args, **kwargs)


class Remat(Module):
    """Gradient checkpointing (`Remat`) wrapper for NNX modules and callables.

    Attributes:
        module_constructor: Wrapped ``Module`` constructor or callable.
        prevent_cse: Whether common subexpression elimination is prevented across checkpoint boundaries.
        policy: Optional checkpointing policy callable or descriptor.
        module_init_args: Positional arguments used to initialize ``module_constructor``.
        module_init_kwargs: Keyword arguments used to initialize ``module_constructor``.
        wrapped_module: Instantiated child ``Module`` (if applicable).
    """

    module_constructor: ModuleTarget
    prevent_cse: bool
    policy: Optional[Union[PolicyCallableProtocol, str]]
    module_init_args: InitArgsType
    module_init_kwargs: InitKwargsType
    wrapped_module: WrappedCallableType

    def __init__(
        self,
        module_constructor: ModuleTarget,
        prevent_cse: bool = True,
        policy: Optional[Union[PolicyCallableProtocol, str]] = None,
        module_init_args: InitArgsType = None,
        module_init_kwargs: InitKwargsType = None,
        **kwargs: MetadataValue,
    ) -> None:
        """Initialize a Remat gradient-checkpointing wrapper.

        Args:
            module_constructor: ``Module`` instance, class, or factory callable to wrap.
            prevent_cse: Whether to prevent CSE optimizations across rematerialization boundaries.
            policy: Optional checkpointing policy.
            module_init_args: Optional positional initialization arguments.
            module_init_kwargs: Optional keyword initialization arguments.
            **kwargs: Additional static configuration metadata.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
        self.prevent_cse = bool(prevent_cse)
        self.policy = policy
        self.module_init_args = module_init_args
        self.module_init_kwargs = module_init_kwargs
        inst = _instantiate_target(module_constructor, module_init_args, module_init_kwargs)
        if isinstance(inst, Module):
            self.wrapped_module = self.register_child("wrapped_module", inst)
        else:
            self.wrapped_module = inst
        self.finish_init()

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Execute the checkpointed module or callable on ``*args`` and ``**kwargs``.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Output ``JaxArray`` or tuple of ``JaxArray``s of the wrapped module computation.
        """
        return _call_wrapped(self.wrapped_module, *args, **kwargs)


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

    def __call__(self, *args: ArrayLike, **kwargs: ArrayLike) -> TransformOutput:
        """Vectorize execution of ``self.wrapped_module`` across ``in_axes``.

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

        step_results: list[TransformOutput] = []
        for idx in range(batch_len):
            step_args: list[ArrayLike] = []
            for arr_a, ax_a in zip(arr_args, arg_axes):
                if ax_a is not None and arr_a.ndim > 0:
                    step_args.append(_slice_along_axis(arr_a, ax_a, idx))
                else:
                    step_args.append(arr_a)
            step_results.append(_call_wrapped(self.wrapped_module, *step_args, **kwargs))

        return _stack_step_results(step_results, self.out_axes)


class Scan(Module):
    """Stateful sequential scan (`Scan`) wrapper unrolling a module across a sequence axis.

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
        *args: ArrayLike,
        **kwargs: ArrayLike,
    ) -> Union[JaxArray, Tuple[JaxArray, ...]]:
        """Execute the scanned loop over ``*args``.

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
            indices = list(range(seq_len - 1, -1, -1)) if self.reverse else list(range(seq_len))
            step_outs: list[JaxArray] = []
            for idx in indices:
                step_x = _slice_along_axis(xs, ax, idx)
                step_res = _call_wrapped(self.wrapped_module, step_x, **kwargs)
                step_y = step_res[0] if isinstance(step_res, tuple) else step_res
                step_outs.append(jnp.expand_dims(step_y, axis=ax))
            return jnp.concatenate(step_outs, axis=ax)

        carry_idx = self.carry_argnum if 0 <= self.carry_argnum < len(args) else 0
        carry = ensure_array(args[carry_idx])
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
            empty_shape = list(carry.shape)
            empty_shape.insert(out_ax, 0)
            c_dt = sanitize_dtype_str(str(carry.dtype).lower())
            empty_out: JaxArray = jnp.zeros(tuple(empty_shape), dtype=c_dt)
            if self.scan_output:
                return carry, empty_out
            return carry

        indices = list(range(seq_len - 1, -1, -1)) if self.reverse else list(range(seq_len))
        ys: list[JaxArray] = []

        accepts_two = False
        if _is_callable_target(self.wrapped_module):
            _, accepts_two = _count_required_positional_params(self.wrapped_module)

        for idx in indices:
            sliced_xs = [_slice_along_axis(arr_s, ax, idx) for arr_s in scanned_arrays]
            step_x = sliced_xs[0]
            if _is_callable_target(self.wrapped_module):
                if accepts_two:
                    call_args: list[JaxArray] = list(sliced_xs)
                    call_args.insert(carry_idx, carry)
                    raw_out = self.wrapped_module(*call_args, **kwargs)
                else:
                    raw_out = self.wrapped_module(step_x, **kwargs)
            else:
                raw_out = step_x
            if isinstance(raw_out, tuple) and len(raw_out) == 2:
                carry = ensure_array(raw_out[0])
                step_y = ensure_array(raw_out[1])
            elif not isinstance(raw_out, (Module, tuple)):
                carry = ensure_array(raw_out)
                step_y = carry
            else:
                step_y = carry
            ys.append(jnp.expand_dims(step_y, axis=out_ax))

        stacked = jnp.concatenate(ys, axis=out_ax)
        if self.scan_output:
            return carry, stacked
        return carry


class Pmap(Module):
    """Parallel map (`Pmap`) wrapper executing an NNX module across parallel devices.

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
        """Execute ``self.wrapped_module`` across the leading device axis of ``*args``.

        Args:
            *args: Positional device-sharded input arrays.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Device-mapped output ``JaxArray`` or tuple of ``JaxArray``s.
        """
        if len(args) == 0 and self.axis_size is None:
            return _call_wrapped(self.wrapped_module, **kwargs)

        arr_args = [ensure_array(a) for a in args]
        num_devices: Optional[int] = self.axis_size
        if num_devices is None:
            for arg_idx, arr_a in enumerate(arr_args):
                if arg_idx not in self.static_broadcasted_argnums and arr_a.ndim > 0:
                    norm_ax = self.in_axes if self.in_axes >= 0 else arr_a.ndim + self.in_axes
                    num_devices = arr_a.shape[norm_ax]
                    break
        if num_devices is None:
            return _call_wrapped(self.wrapped_module, *args, **kwargs)

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
