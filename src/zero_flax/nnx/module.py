"""Module system for nnx."""

from typing import Any
from zero_flax.nnx.state import Variable, State


class Module:
    def __init__(self):
        super().__setattr__("_is_initializing", True)
        super().__setattr__("_children", {})
        super().__setattr__("_variables", {})

    def __setattr__(self, name: str, value: Any):
        if name in ["_is_initializing", "_children", "_variables"]:
            super().__setattr__(name, value)
            return

        if not getattr(self, "_is_initializing", True):
            if not hasattr(self, name):
                raise ValueError(
                    f"Cannot add new attribute {name} after initialization"
                )

        if isinstance(value, Module):
            if not hasattr(self, "_children"):
                super().__setattr__("_children", {})
            self._children[name] = value
        elif isinstance(value, Variable):
            if not hasattr(self, "_variables"):
                super().__setattr__("_variables", {})
            self._variables[name] = value

        super().__setattr__(name, value)

    def state(self) -> State:
        s = State()
        if hasattr(self, "_variables"):
            for k, v in self._variables.items():
                s[k] = v
        if hasattr(self, "_children"):
            for k, child in self._children.items():
                s[k] = child.state()
        return s

    def update(self, state: State):
        for k, v in state.items():
            if hasattr(self, "_children") and k in self._children:
                self._children[k].update(v)
            elif hasattr(self, "_variables") and k in self._variables:
                self._variables[k].value = v.value
