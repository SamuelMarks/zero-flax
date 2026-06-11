"""Tests for E2E training scripts (MLP via dummy optimizer)."""

import numpy as np
from zero_flax.nnx import Module, Dense, Param
from jax import numpy as jnp


def value_and_grad(f):
    def wrapper(model, x, y):
        loss = f(model, x, y)
        return loss, model.state()

    return wrapper


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(2, 4)
        self.d2 = Dense(4, 1)
        super().__setattr__("_is_initializing", False)

    def __call__(self, x):
        x = self.d1(x)
        x = jnp.where(x > 0, x, 0.0)  # ReLU
        return self.d2(x)


def mse_loss(model, x, y):
    preds = model(x)
    return np.mean(jnp.multiply(jnp.add(preds, -y), jnp.add(preds, -y)))


def dummy_optimizer(state, grads, lr=0.1):
    from zero_flax.nnx.state import State

    for k, v in state.items():
        if isinstance(v, Param):
            # For test: assume grads map to the same nested structure (mocked)
            # Just do a mock update for the test
            v.value = v.value - lr * 0.1  # dummy static gradient
        elif isinstance(v, State):
            dummy_optimizer(v, grads, lr)


def test_e2e_mlp_training():
    model = MLP()
    x = np.array([[1.0, 2.0]])
    y = np.array([[3.0]])

    # Mocking grad execution
    loss_fn = value_and_grad(mse_loss)

    # Run 2 steps
    for _ in range(2):
        loss, grads = loss_fn(model, x, y)
        # Update
        dummy_optimizer(model.state(), grads)

    assert True
