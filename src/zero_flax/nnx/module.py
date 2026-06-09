"""Module system for nnx."""

from typing import Any  # noqa: F401
from zero_flax.nnx.state import Variable, State  # noqa: F401


class Module:
    """Base class for nnx Modules."""

    def __init__(self):
        super().__setattr__("_is_initializing", True)

    def __setattr__(self, name: str, value: Any):
        if getattr(self, "_is_initializing", False) or isinstance(
            value, (Variable, Module)
        ):
            super().__setattr__(name, value)
        else:
            raise ValueError(
                f"Cannot mutate non-Variable attribute '{name}' after initialization."
            )

    def state(self) -> State:
        """Extracts the state of the module."""
        state = State()
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, Variable):
                state[name] = value
            elif isinstance(value, Module):
                state[name] = value.state()
        return state

    def update(self, state: State):
        """Updates the module's state from a State object."""
        for name, value in state.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                if isinstance(attr, Variable):
                    attr.value = value.value
                elif isinstance(attr, Module):
                    attr.update(value)
