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
