"""End-to-end 5-repository ecosystem training, gradient, and checkpoint recovery tests for zero-flax."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import zero_chex
import zero_grain as grain
import zero_optax
import zero_orbax.checkpoint as ocp
from zero_jax import numpy as jnp
from zero_jax.nn import initializers

from zero_flax.nnx import Dense, Jit, Module, Optimizer, Param, State, Vmap
from zero_flax.types import ArrayLike, JaxArray, ensure_array


class MLP(Module):
    """Two-layer Multi-Layer Perceptron with ReLU hidden activation.

    Attributes:
        d1: First ``Dense`` layer mapping ``2 -> 4``.
        d2: Second ``Dense`` layer mapping ``4 -> 1``.
    """

    d1: Dense
    d2: Dense

    def __init__(self) -> None:
        """Initialize the two-layer MLP with deterministic ones weights."""
        super().__init__()
        self.d1 = self.register_child(
            "d1",
            Dense(
                2,
                4,
                kernel_init=initializers.ones,
                bias_init=initializers.zeros,
            ),
        )
        self.d2 = self.register_child(
            "d2",
            Dense(
                4,
                1,
                kernel_init=initializers.ones,
                bias_init=initializers.zeros,
            ),
        )
        self.finish_init()

    def __call__(self, x: ArrayLike) -> JaxArray:
        """Evaluate the MLP forward pass on ``x``.

        Args:
            x: Input array of shape ``(batch, 2)``.

        Returns:
            Output predictions of shape ``(batch, 1)``.
        """
        hidden = jnp.maximum(self.d1(x), 0.0)
        return self.d2(hidden)


def mse_loss(model: MLP, x: ArrayLike, y: ArrayLike) -> JaxArray:
    """Compute Mean Squared Error loss between ``model(x)`` and targets ``y``.

    Args:
        model: ``MLP`` instance to evaluate.
        x: Input feature batch.
        y: Target value batch.

    Returns:
        Scalar MSE loss ``JaxArray``.
    """
    preds = model(x)
    diff = preds - ensure_array(y)
    res: JaxArray = jnp.mean(diff * diff)
    return res


def value_and_grad(
    loss_fn: Callable[[MLP, ArrayLike, ArrayLike], JaxArray],
    eps: float = 1e-3,
) -> Callable[[MLP, ArrayLike, ArrayLike], tuple[JaxArray, State]]:
    """Construct a central finite-difference gradient evaluator for ``loss_fn``.

    Perturbs every scalar element of every ``Param`` in ``model.state()`` by ``+/- eps``
    and evaluates ``(loss_fn(theta + eps) - loss_fn(theta - eps)) / (2 * eps)``.

    Args:
        loss_fn: Loss function accepting ``(model, x, y)``.
        eps: Finite-difference step size. Defaults to ``1e-3``.

    Returns:
        Wrapper returning ``(loss_value, grads_state)``.
    """

    def wrapper(model: MLP, x: ArrayLike, y: ArrayLike) -> tuple[JaxArray, State]:
        """Evaluate ``loss_fn`` and return exact central finite-difference parameter gradients.

        Args:
            model: ``MLP`` model instance.
            x: Input batch.
            y: Target batch.

        Returns:
            Pair ``(loss, grads_state)``.
        """
        base_loss = loss_fn(model, x, y)
        grads = State()
        for key, sub_state in model.state().items():
            if isinstance(sub_state, State):
                layer_grads = State()
                for p_name, var in sub_state.items():
                    if isinstance(var, Param):
                        orig_np = np.asarray(ensure_array(var.value), dtype=np.float64).copy()
                        grad_np = np.zeros_like(orig_np, dtype=np.float64)
                        it = np.ndindex(orig_np.shape)
                        for idx in it:
                            pert_plus = orig_np.copy()
                            pert_plus[idx] += eps
                            var.value = ensure_array(pert_plus.astype(np.float32))
                            loss_plus = float(np.asarray(loss_fn(model, x, y)))

                            pert_minus = orig_np.copy()
                            pert_minus[idx] -= eps
                            var.value = ensure_array(pert_minus.astype(np.float32))
                            loss_minus = float(np.asarray(loss_fn(model, x, y)))

                            grad_np[idx] = (loss_plus - loss_minus) / (2.0 * eps)
                        var.value = ensure_array(orig_np.astype(np.float32))
                        layer_grads[p_name] = Param(ensure_array(grad_np.astype(np.float32)))
                grads[key] = layer_grads
        return base_loss, grads

    return wrapper


def compute_analytical_mlp_grads(model: MLP, x: ArrayLike, y: ArrayLike) -> State:
    """Compute exact analytical backpropagation gradients for ``MLP`` under ``mse_loss``.

    Args:
        model: ``MLP`` instance.
        x: Input batch of shape ``(N, 2)``.
        y: Target batch of shape ``(N, 1)``.

    Returns:
        Analytical gradient ``State`` matching ``model.state()``.
    """
    x_np = np.asarray(ensure_array(x), dtype=np.float32)
    y_np = np.asarray(ensure_array(y), dtype=np.float32)
    w1 = np.asarray(ensure_array(model.d1.kernel.value), dtype=np.float32)
    b1 = (
        np.asarray(ensure_array(model.d1.bias.value), dtype=np.float32)
        if model.d1.bias is not None
        else np.zeros((4,), dtype=np.float32)
    )
    w2 = np.asarray(ensure_array(model.d2.kernel.value), dtype=np.float32)
    b2 = (
        np.asarray(ensure_array(model.d2.bias.value), dtype=np.float32)
        if model.d2.bias is not None
        else np.zeros((1,), dtype=np.float32)
    )

    z1 = x_np @ w1 + b1
    h1 = np.maximum(z1, 0.0)
    preds = h1 @ w2 + b2
    n_elems = float(preds.size)

    d_preds = (2.0 / n_elems) * (preds - y_np)
    d_w2 = h1.T @ d_preds
    d_b2 = np.sum(d_preds, axis=0)
    d_h1 = d_preds @ w2.T
    d_z1 = d_h1 * (z1 > 0.0).astype(np.float32)
    d_w1 = x_np.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0)

    return State(
        {
            "d1": State(
                {
                    "kernel": Param(ensure_array(d_w1)),
                    "bias": Param(ensure_array(d_b1)),
                }
            ),
            "d2": State(
                {
                    "kernel": Param(ensure_array(d_w2)),
                    "bias": Param(ensure_array(d_b2)),
                }
            ),
        }
    )


def apply_sgd_step(state: State, grads: State, lr: float = 0.02) -> None:
    """Apply an in-place stochastic gradient descent update to ``state``.

    Args:
        state: Hierarchical model ``State`` to update.
        grads: Hierarchical gradient ``State``.
        lr: Learning rate step size.
    """
    for key, val in grads.items():
        target = state.get(key)
        if isinstance(val, State) and isinstance(target, State):
            apply_sgd_step(target, val, lr=lr)
        elif isinstance(val, Param) and isinstance(target, Param):
            target.value = ensure_array(target.value) - lr * ensure_array(val.value)


def test_mlp_training() -> None:
    """Verify analytical gradient parity and multi-step SGD convergence on non-zero targets."""
    model = MLP()
    x = jnp.array([[1.0, 0.5], [0.5, 1.5], [2.0, 1.0], [1.5, 0.5]])
    y = jnp.array([[2.0], [3.0], [4.0], [2.5]])

    step_fn = value_and_grad(mse_loss, eps=1e-3)
    initial_loss, num_grads = step_fn(model, x, y)
    ana_grads = compute_analytical_mlp_grads(model, x, y)

    for layer_key in ("d1", "d2"):
        num_layer = num_grads[layer_key]
        ana_layer = ana_grads[layer_key]
        assert isinstance(num_layer, State) and isinstance(ana_layer, State)
        for param_key in ("kernel", "bias"):
            num_p = num_layer[param_key]
            ana_p = ana_layer[param_key]
            assert isinstance(num_p, Param) and isinstance(ana_p, Param)
            np.testing.assert_allclose(
                np.asarray(num_p.value),
                np.asarray(ana_p.value),
                rtol=1e-2,
                atol=1e-2,
            )

    prev_loss = float(np.asarray(initial_loss))
    for _ in range(12):
        grads = compute_analytical_mlp_grads(model, x, y)
        st = model.state()
        apply_sgd_step(st, grads, lr=0.01)
        model.update(st)
        cur_loss = float(np.asarray(mse_loss(model, x, y)))
        assert cur_loss < prev_loss
        prev_loss = cur_loss

    assert prev_loss < 0.2 * float(np.asarray(initial_loss))


def test_full_five_repo_ecosystem_e2e_pipeline(tmp_path: Path) -> None:
    """Verify the full 5-repository pipeline: zero-grain -> zero-flax -> zero-optax -> zero-chex -> zero-orbax.

    Args:
        tmp_path: Pytest temporary directory fixture for checkpoint storage.
    """
    raw_records: list[dict[str, list[float]]] = [
        {"x": [1.0, 0.5], "y": [2.0]},
        {"x": [0.5, 1.5], "y": [3.0]},
        {"x": [2.0, 1.0], "y": [4.0]},
        {"x": [1.5, 0.5], "y": [2.5]},
        {"x": [1.0, 1.0], "y": [2.5]},
        {"x": [0.5, 0.5], "y": [1.5]},
        {"x": [2.0, 0.5], "y": [3.0]},
        {"x": [0.5, 2.0], "y": [3.5]},
    ]
    source = grain.InMemoryDataSource(raw_records)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=3,
        shard_options=grain.NoSharding(),
        shuffle=False,
        seed=0,
    )
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=2, drop_remainder=True)],
    )
    data_iter = iter(loader)

    model = MLP()
    jit_model = Jit(model)
    vmap_d1 = Vmap(model.d1, in_axes=0, out_axes=0)

    schedule = zero_optax.linear_schedule(
        init_value=0.02,
        end_value=0.01,
        transition_steps=10,
    )
    tx = zero_optax.chain(
        zero_optax.clip_by_global_norm(2.0),
        zero_optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )
    optimizer = Optimizer(model, tx)

    ckpt_dir = tmp_path / "e2e_composite_ckpts"
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    manager = ocp.CheckpointManager(
        ckpt_dir,
        checkpointers={
            "model": ocp.StandardCheckpointer(),
            "opt": ocp.PyTreeCheckpointer(),
            "data": ocp.Checkpointer(grain.PyGrainCheckpointHandler()),
        },
        options=options,
    )

    for step in range(1, 5):
        batch = next(data_iter)
        assert isinstance(batch, dict)
        bx = ensure_array(batch["x"])
        by = jnp.reshape(ensure_array(batch["y"]), (-1, 1))

        zero_chex.assert_rank(bx, 2)
        zero_chex.assert_shape(bx, (2, 2))
        zero_chex.assert_shape(by, (2, 1))

        vmapped_h = vmap_d1(bx)
        zero_chex.assert_shape(vmapped_h, (2, 4))

        preds = jit_model(bx)
        zero_chex.assert_shape(preds, (2, 1))
        optax_loss = zero_optax.losses.squared_error(
            preds,
            by,
        )
        assert optax_loss.shape == (2, 1)

        grads = compute_analytical_mlp_grads(model, bx, by)
        zero_chex.assert_trees_all_equal_shapes(model.state(), grads)
        optimizer.update(grads)

        if step == 2:
            manager.save(
                step,
                items={
                    "model": model.state().to_pure_dict(include_variable_metadata=True),
                    "opt": optimizer.state_dict(),
                    "data": data_iter,
                },
            )

    state_at_step_4 = model.state()
    next_batch_from_orig = next(data_iter)

    restored_model = MLP()
    restored_opt = Optimizer(restored_model, tx)
    restored_iter = iter(loader)

    restored_bundle = manager.restore(
        2,
        items={
            "model": restored_model.state().to_pure_dict(include_variable_metadata=True),
            "opt": restored_opt.state_dict(),
            "data": restored_iter,
        },
    )
    assert isinstance(restored_bundle, dict)
    if isinstance(restored_bundle.get("opt"), dict):
        restored_opt.load_state_dict(restored_bundle["opt"])

    for _ in range(3, 5):
        rbatch = next(restored_iter)
        assert isinstance(rbatch, dict)
        rbx = ensure_array(rbatch["x"])
        rby = jnp.reshape(ensure_array(rbatch["y"]), (-1, 1))
        rgrads = compute_analytical_mlp_grads(restored_model, rbx, rby)
        restored_opt.update(rgrads)

    zero_chex.assert_trees_all_close(restored_model.state(), state_at_step_4)
    next_batch_from_restored = next(restored_iter)
    assert isinstance(next_batch_from_orig, dict) and isinstance(next_batch_from_restored, dict)
    zero_chex.assert_trees_all_close(
        ensure_array(next_batch_from_restored["x"]),
        ensure_array(next_batch_from_orig["x"]),
    )
    zero_chex.assert_trees_all_close(
        ensure_array(next_batch_from_restored["y"]),
        ensure_array(next_batch_from_orig["y"]),
    )
    manager.close()


def test_six_repo_shard_rngs_jit_vmap_remat_grad_optax_chex_composite_orbax(
    tmp_path: Path,
) -> None:
    """Verify 6-repo pipeline: Grain ShardByJaxProcess + Rngs -> Jit/Vmap/Remat + value_and_grad -> Optax -> Chex -> composite save_state/restore_state.

    Args:
        tmp_path: Pytest temporary directory fixture for composite checkpoint storage.
    """
    import zero_jax

    from zero_flax.nnx import Remat, Rngs, merge, restore_state, save_state, split
    from zero_flax.types import GrainBatchAdapter

    rngs = Rngs(42)
    assert 0.0 <= rngs.random() < 1.0
    assert -1.0 <= rngs.uniform(-1.0, 1.0) < 1.0
    assert 0 <= rngs.integers(5) < 5
    assert rngs.integers(3, 3) == 3
    assert sorted(rngs.permutation(4)) == [0, 1, 2, 3]
    assert sorted(rngs.permutation([10, 20, 30])) == [10, 20, 30]

    records = [
        grain.Record(grain.RecordMetadata(index=i), {"x": [float(i + 1), 0.5], "y": [float(i + 2)]})
        for i in range(6)
    ]
    adapter = GrainBatchAdapter(feature_key="x", target_key="y", dtype="float32")
    rec_x, rec_y = adapter.to_feature_target_pair(records[0])
    assert rec_x.shape == (2,) and rec_y.shape == (1,)
    assert adapter.to_array(records[0]).shape == (2,)
    assert "x" in adapter.to_array_dict(records[0])

    raw_rows: list[dict[str, list[float]]] = [
        {"x": [float(i + 1), 0.5], "y": [float(i + 2)]} for i in range(8)
    ]
    sampler = grain.IndexSampler(
        num_records=len(raw_rows),
        num_epochs=2,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        seed=rngs.integers(1, 1000),
    )
    loader = grain.DataLoader(
        data_source=grain.InMemoryDataSource(raw_rows),
        sampler=sampler,
        operations=[grain.Batch(batch_size=2, drop_remainder=True)],
    )
    data_iter = iter(loader)

    model = MLP()
    jit_mod = Jit(model)
    remat_d1 = Remat(model.d1, prevent_cse=True, static_argnums=())
    vmap_d2 = Vmap(model.d2, in_axes=0, out_axes=0)

    traced_stage = jit_mod.trace(jnp.ones((2, 2)))
    lowered_stage = jit_mod.lower(jnp.ones((2, 2)))
    assert traced_stage is not None and lowered_stage is not None

    hp_factory = zero_optax.inject_hyperparams(zero_optax.adamw)
    assert callable(hp_factory)
    tx = hp_factory(0.01)
    optimizer = Optimizer(model, tx)
    lr_val = optimizer.get_hyperparam("learning_rate")
    assert lr_val is not None
    assert optimizer.get_hyperparam("nonexistent_hp") is None

    @zero_chex.assert_max_traces(n=2)
    def _traced_forward(bx_in: JaxArray) -> JaxArray:
        """Execute forward pass through ``jit_mod`` under ``assert_max_traces``.

        Args:
            bx_in: Batch input array.

        Returns:
            Model predictions.
        """
        out_p = jit_mod(bx_in)
        return out_p if isinstance(out_p, jnp.ndarray) else out_p[0]

    gdef, _ = split(model)

    def _loss_on_state(st: State, bx_in: JaxArray, by_in: JaxArray) -> JaxArray:
        """Compute scalar MSE loss for ``zero_jax.value_and_grad`` over ``State``.

        Args:
            st: Model ``State`` PyTree.
            bx_in: Input batch features.
            by_in: Target batch labels.

        Returns:
            Scalar mean squared error loss.
        """
        m_cur = merge(gdef, st)
        preds_cur = m_cur(bx_in)
        diff = preds_cur - by_in
        return jnp.mean(diff * diff)

    vg_fn = zero_jax.value_and_grad(_loss_on_state)

    ckpt_path = tmp_path / "composite_save_state"
    for step in range(1, 4):
        batch = next(data_iter)
        assert isinstance(batch, dict)
        bx, by_1d = adapter.to_feature_target_pair(batch)
        by = jnp.reshape(by_1d, (-1, 1))

        h_remat = remat_d1(bx)
        out_vmap = vmap_d2(jnp.tanh(h_remat))
        zero_chex.assert_trees_all_finite(out_vmap)

        preds = _traced_forward(bx)
        zero_chex.assert_trees_all_finite(preds)

        loss_val, state_grads = vg_fn(model.state(), bx, by)
        zero_chex.assert_trees_all_finite(loss_val)
        zero_chex.assert_trees_all_finite(state_grads)
        optimizer.update(state_grads)

        if step == 2:
            assert save_state(
                ckpt_path,
                model.state(),
                force=True,
                optimizer=optimizer,
                dataloader_iterator=data_iter,
            )
            assert save_state(
                ckpt_path,
                model.state(),
                force=True,
                optimizer=optimizer,
                dataloader_iterator=data_iter,
            )

    assert jit_mod.trace_count >= 1
    final_state = model.state()
    next_batch_expected = next(data_iter)

    restored_model = MLP()
    restored_opt = Optimizer(restored_model, tx)
    restored_iter = iter(loader)
    restored_st = restore_state(
        ckpt_path,
        target=restored_model.state(),
        optimizer=restored_opt,
        dataloader_iterator=restored_iter,
    )
    restored_model.update(restored_st)

    rbatch = next(restored_iter)
    assert isinstance(rbatch, dict)
    rbx, rby_1d = adapter.to_feature_target_pair(rbatch)
    rby = jnp.reshape(rby_1d, (-1, 1))
    _, r_grads = vg_fn(restored_model.state(), rbx, rby)
    restored_opt.update(r_grads)

    zero_chex.assert_trees_all_close(restored_model.state(), final_state)
    next_batch_actual = next(restored_iter)
    assert isinstance(next_batch_expected, dict) and isinstance(next_batch_actual, dict)
    zero_chex.assert_trees_all_close(
        adapter.to_array(next_batch_actual),
        adapter.to_array(next_batch_expected),
    )


def test_scan_and_rnn_lstm_gru_lax_scan_clip_orbax_roundtrip(tmp_path: Path) -> None:
    """Verify Scan and RNN (LSTMCell, GRUCell) backed by zero_jax.lax.scan with clip_by_global_norm and Orbax restore.

    Args:
        tmp_path: Pytest temporary directory fixture for checkpoint storage.
    """
    from zero_flax.nnx import RNN, GRUCell, LSTMCell, Rngs, Scan, restore_state, save_state

    rngs = Rngs(7)
    lstm_cell = LSTMCell(3, 4, rngs=rngs)
    gru_cell = GRUCell(4, 2, rngs=rngs)
    rnn_lstm = RNN(lstm_cell, return_carry=True, rngs=rngs)
    rnn_rev = RNN(gru_cell, reverse=True, keep_order=False, return_carry=False, rngs=rngs)
    scan_gru = Scan(gru_cell, in_axes=1, out_axes=1)

    seq_x = jnp.ones((2, 3, 3))
    init_c = lstm_cell.initialize_carry((2, 3), rngs=rngs)
    carry_out, lstm_seq = rnn_lstm(seq_x, initial_carry=init_c)
    assert isinstance(carry_out, tuple) and lstm_seq.shape == (2, 3, 4)

    rev_out = rnn_rev(lstm_seq)
    assert isinstance(rev_out, jnp.ndarray) and rev_out.shape == (2, 3, 2)

    gru_c0 = gru_cell.initialize_carry((2, 4), rngs=rngs)
    scan_res = scan_gru(gru_c0, lstm_seq)
    assert isinstance(scan_res, tuple) and len(scan_res) == 2

    tx = zero_optax.chain(
        zero_optax.clip_by_global_norm(1.0),
        zero_optax.adam(learning_rate=1e-2),
    )
    opt = Optimizer(gru_cell, tx)
    grads = gru_cell.state()
    opt.update(grads)

    ckpt_dir = tmp_path / "rnn_scan_ckpt"
    assert save_state(ckpt_dir, gru_cell.state(), optimizer=opt)
    restored_gru = GRUCell(4, 2, rngs=Rngs(99))
    restored_opt = Optimizer(restored_gru, tx)
    restored_state = restore_state(ckpt_dir, optimizer=restored_opt)
    restored_gru.update(restored_state)
    zero_chex.assert_trees_all_close(restored_gru.state(), gru_cell.state())

    dl_only_dir = tmp_path / "dl_only_ckpt"
    simple_loader = grain.DataLoader(
        data_source=grain.InMemoryDataSource([{"x": [1.0]}]),
        sampler=grain.IndexSampler(num_records=1, num_epochs=1, shard_options=grain.NoSharding()),
    )
    dl_iter = iter(simple_loader)
    assert save_state(dl_only_dir, gru_cell.state(), force=False, dataloader_iterator=dl_iter)
    dl_iter_restore = iter(simple_loader)
    dl_restored_st = restore_state(dl_only_dir, dataloader_iterator=dl_iter_restore)
    assert len(dl_restored_st) > 0


def test_pytree_sharding_and_transform_edge_cases() -> None:
    """Verify 100% line and branch coverage on PyTree methods, sharding accessors, GrainBatchAdapter, and transforms."""
    import pytest
    import zero_jax
    from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
    from ml_switcheroo_compiler.core.tensor import TensorConfig
    from ml_switcheroo_ir import LogicalMesh, PartitionSpec
    from zero_jax.sharding import NamedSharding, SingleDeviceSharding

    from zero_flax.core import FrozenDict, register_frozendict_pytree
    from zero_flax.nnx import BatchStat, Jit, Param, RngCount, RngKey, Vmap
    from zero_flax.nnx.state import register_state_pytree
    from zero_flax.nnx.variables import VariableBase, register_variable_pytrees
    from zero_flax.types import GrainBatchAdapter

    assert register_frozendict_pytree() is FrozenDict
    assert register_state_pytree() is State
    assert len(register_variable_pytrees()) >= 10

    fd = FrozenDict({"b": 2, "a": jnp.array([1.0])})
    assert fd.chex_mappable_fields() == ("a", "b")
    assert len(fd.chex_field_values()) == 2
    fd_k, fd_aux = fd.tree_flatten_with_keys()
    fd_p, _ = fd.tree_flatten()
    assert len(fd_k) == 2 and len(fd_p) == 2
    fd_rebuilt = FrozenDict.tree_unflatten(fd_aux, [jnp.array([3.0]), {"nested": [1.0, 2.0]}])
    assert isinstance(fd_rebuilt["b"], FrozenDict)
    assert FrozenDict.tree_unflatten(("x",), [None])["x"] is None

    st = State({"b": BatchStat(jnp.array([2.0])), "a": Param(jnp.array([1.0]))})
    assert "a" in st.to_chex_tree()
    assert st.chex_mappable_fields() == ("a", "b")
    assert len(st.chex_field_values()) == 2
    st_k, st_aux = st.tree_flatten_with_keys()
    st_p, _ = st.tree_flatten()
    assert len(st_k) == 2 and len(st_p) == 2
    st_rebuilt = State.tree_unflatten(st_aux, [jnp.array([5.0]), {"sub": jnp.array([6.0])}])
    assert isinstance(st_rebuilt["b"], State)

    vb = VariableBase(tag="base")
    assert vb.chex_mappable_fields() == ("value",)
    assert vb.chex_field_values() == (None,)

    mesh = LogicalMesh(shape={"x": 1})
    named_sh = NamedSharding(mesh, PartitionSpec(("x",)))
    single_sh = SingleDeviceSharding("cpu:0")
    pspec = PartitionSpec(("x",))

    p_str = Param(jnp.array([1.0, 2.0]), sharding="x")
    assert isinstance(p_str.sharding, PartitionSpec)
    p_tup = Param(jnp.array([1.0, 2.0]), sharding=("x", None))
    assert isinstance(p_tup.sharding, PartitionSpec)
    p_int_sh = Param(10, sharding=123)
    assert p_int_sh.sharding == 123
    p_none_sh = Param(10)
    assert p_none_sh.sharding is None
    p_arr_default = Param(jnp.array([1.0]))
    assert p_arr_default.sharding is not None

    p_str.set_sharding(named_sh)
    assert p_str.sharding is named_sh
    p_str.set_sharding(pspec)
    assert p_str.sharding is pspec
    p_str.set_sharding(single_sh)
    assert p_str.sharding is single_sh
    p_str.set_sharding("batch")
    assert isinstance(p_str.sharding, PartitionSpec)
    p_str.set_sharding(999)
    p_int_sh.set_sharding("x")
    p_rep = p_int_sh.replace(20)
    p_cp = p_int_sh.copy()
    assert isinstance(p_rep.sharding, PartitionSpec) and isinstance(p_cp.sharding, PartitionSpec)

    assert p_str.chex_field_values() == (p_str.value,)
    vk, vaux = p_str.tree_flatten_with_keys()
    vp, _ = p_str.tree_flatten()
    assert len(vk) == 1 and len(vp) == 1
    ct_raw = CompilerTensor([1.0, 2.0], TensorConfig(shape=(2,), dtype="float32"))
    assert isinstance(Param.tree_unflatten({"sharding": ("x",)}, [ct_raw]), Param)
    assert isinstance(Param.tree_unflatten("not_a_dict", [[1.0, 2.0]]), Param)
    assert isinstance(Param.tree_unflatten({}, []), Param)
    assert isinstance(RngCount.tree_unflatten({}, [5]), RngCount)
    assert isinstance(RngKey.tree_unflatten({}, [jnp.array([0, 1])]), RngKey)

    adapter = GrainBatchAdapter(feature_key="feat", target_key="lbl")
    assert adapter.to_array({"other": [1.0, 2.0]}).shape == (2,)
    assert adapter.to_array([1.0, 2.0, 3.0]).shape == (3,)
    assert "feat" in adapter.to_array_dict([1.0, 2.0])
    with pytest.raises(KeyError, match="requires a mapping batch"):
        adapter.to_feature_target_pair([1.0, 2.0])

    def _fn_pair(
        *args: ArrayLike | tuple[ArrayLike, ...],
        **_kwargs: ArrayLike,
    ) -> JaxArray | tuple[JaxArray | tuple[JaxArray, ...], ...]:
        """Return nested tuple output for _coerce_transform_output coverage.

        Args:
            *args: Input array arguments.
            **_kwargs: Unused keyword arguments.

        Returns:
            Pair of array and nested tuple of arrays.
        """
        first_arg = args[0] if len(args) > 0 and not isinstance(args[0], tuple) else 1.0
        arr = ensure_array(first_arg)
        return arr, (arr, arr)

    j_fn = Jit(
        _fn_pair,
        in_axis_resources=single_sh,
        out_axis_resources=[single_sh, None],
    )
    assert j_fn.trace(jnp.ones((2,))) is not None
    assert j_fn.lower(jnp.ones((2,))) is not None
    pair_out = j_fn(jnp.ones((2,)))
    assert isinstance(pair_out, tuple)

    mlp = MLP()
    j_mlp = Jit(mlp)

    def _outer_traced(x_in: JaxArray) -> JaxArray:
        """Call ``j_mlp`` and ``j_fn`` while outer ``zero_jax.grad`` tracing is active.

        Args:
            x_in: Input tensor.

        Returns:
            Scalar sum output.
        """
        y_m = j_mlp(x_in)
        y_f = j_fn(x_in)
        y_arr = y_m if isinstance(y_m, jnp.ndarray) else y_m[0]
        f_arr = y_f[0] if isinstance(y_f, tuple) else y_f
        return jnp.sum(y_arr) + jnp.sum(f_arr)

    grad_outer = zero_jax.grad(_outer_traced)(jnp.ones((2, 2)))
    assert grad_outer.shape == (2, 2)

    vmap_with_var_axes = Vmap(
        mlp,
        in_axes=0,
        out_axes=0,
        variable_axes={Param: 0},
    )
    vm_out = vmap_with_var_axes(jnp.ones((2, 2)))
    assert isinstance(vm_out, jnp.ndarray) and vm_out.shape == (2, 1)

    from zero_flax.nnx import Module, Remat

    remat_non_callable = Remat(Module())
    with pytest.raises(TypeError, match="not callable"):
        remat_non_callable(jnp.ones((2,)))
