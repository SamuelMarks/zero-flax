"""Module system for nnx."""

from typing import Any
from zero_flax.nnx.state import Variable, State


class Module:
    """Docstring."""

    _children: dict[str, Any]
    _variables: dict[str, Variable]
    _is_initializing: bool

    def __init__(self):
        """Docstring."""
        super().__setattr__("_is_initializing", True)
        super().__setattr__("_children", {})
        super().__setattr__("_variables", {})

    def __setattr__(self, name: str, value: Any):
        """Docstring."""
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
        """Docstring."""
        s = State()
        if hasattr(self, "_variables"):
            for k, v in self._variables.items():
                s[k] = v
        if hasattr(self, "_children"):
            for k, child in self._children.items():
                s[k] = child.state()
        return s

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Docstring."""
        state_val = args[0] if args else kwargs.get("state")
        if state_val is None:
            return
        for k, v in state_val.items():
            if hasattr(self, "_children") and k in self._children:
                self._children[k].update(v)
            elif hasattr(self, "_variables") and k in self._variables:
                self._variables[k].value = v.value


class GraphDef:
    """Docstring."""

    def __init__(self, cls, children_defs, static_fields):
        """Docstring."""
        self.cls = cls
        self.children_defs = children_defs
        self.static_fields = static_fields

    def merge(self, state: State):
        """Docstring."""
        from zero_flax.nnx.state import merge as state_merge

        if isinstance(state, tuple):
            state = state_merge(*state)

        m = self.cls.__new__(self.cls)
        m._is_initializing = True
        m._children = {}
        m._variables = {}
        for k, v in self.static_fields.items():
            setattr(m, k, v)
        for k, child_def in self.children_defs.items():
            child_state = State(
                {ck: cv for ck, cv in state.items() if ck.startswith(k + ".")}
            )
            # Reconstruct child state without prefix
            local_child_state = State(
                {ck[len(k) + 1 :]: cv for ck, cv in child_state.items()}
            )
            m._children[k] = child_def.merge(local_child_state)
            setattr(m, k, m._children[k])
        for k, v in state.items():
            if "." not in k:
                m._variables[k] = v
                setattr(m, k, v)
        m._is_initializing = False
        return m


def split(module, *filters):
    """Docstring."""
    # For dummy implementation in API shell, we return empty state and empty gdef
    # if the module is just standard dict/list containers.
    # We will traverse simply
    children_defs = {}
    static_fields = {}
    state_dict = State()

    def _extract_state(m, prefix=""):
        """Docstring."""
        if hasattr(m, "_variables"):
            for k, v in m._variables.items():
                state_dict[prefix + k] = v
        if hasattr(m, "_children"):
            for k, c in m._children.items():
                _extract_state(c, prefix + k + ".")

    _extract_state(module)
    for k, v in module.__dict__.items():
        if k not in ["_children", "_variables", "_is_initializing"]:
            static_fields[k] = v
    gdef = GraphDef(module.__class__, {}, static_fields)
    if not filters:
        return gdef, state_dict

    # ensure split always returns tuple with length >= 2
    res = state_dict.split(*filters)

    return (gdef,) + res


def merge(*args):
    """Docstring."""
    # If the first arg is a GraphDef, use GraphDef.merge
    # If the first arg is a State/tuple of states, use state.merge
    if args and type(args[0]).__name__ == "GraphDef":
        graphdef = args[0]
        states = args[1:]
        from zero_flax.nnx.state import merge as state_merge

        return graphdef.merge(state_merge(*states))
    else:
        from zero_flax.nnx.state import merge as state_merge

        return state_merge(*args)
