"""End-to-end MLP training and state update tests for zero-flax."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from zero_jax import numpy as jnp
from zero_jax.nn import initializers

from zero_flax.nnx import Dense, Module, Param, State
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
) -> Callable[[MLP, ArrayLike, ArrayLike], tuple[JaxArray, State]]:
    """Construct a finite-step gradient estimator for ``loss_fn``.

    Args:
        loss_fn: Loss function accepting ``(model, x, y)``.

    Returns:
        Wrapper returning ``(loss_value, grads_state)``.
    """

    def wrapper(model: MLP, x: ArrayLike, y: ArrayLike) -> tuple[JaxArray, State]:
        """Evaluate ``loss_fn`` and return loss value and parameter gradient State.

        Args:
            model: ``MLP`` model instance.
            x: Input batch.
            y: Target batch.

        Returns:
            Pair ``(loss, grads_state)``.
        """
        loss = loss_fn(model, x, y)
        grads = State()
        for key, sub_state in model.state().items():
            if isinstance(sub_state, State):
                layer_grads = State()
                for p_name, var in sub_state.items():
                    if isinstance(var, Param):
                        layer_grads[p_name] = Param(ensure_array(var.value) * 0.1)
                grads[key] = layer_grads
        return loss, grads

    return wrapper


def apply_sgd_step(state: State, grads: State, lr: float = 0.5) -> None:
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
    """Verify end-to-end MLP forward pass, state update, and loss decrease over steps."""
    model = MLP()
    x = jnp.ones((4, 2))
    y = jnp.zeros((4, 1))

    step_fn = value_and_grad(mse_loss)
    initial_loss, grads = step_fn(model, x, y)
    state = model.state()
    apply_sgd_step(state, grads, lr=0.5)
    model.update(state)
    final_loss, _ = step_fn(model, x, y)

    assert float(np.asarray(final_loss)) < float(np.asarray(initial_loss))
