"""State and Variable tracking for nnx."""

from typing import Any


class Variable:
    """Docstring."""

    def __init__(self, value: Any):
        """Docstring."""
        self.value = value

    def __repr__(self):
        """Docstring."""
        return f"{self.__class__.__name__}({self.value})"


class Param(Variable):
    """Docstring."""

    pass


class BatchStat(Variable):
    """Docstring."""

    pass


class Rng(Variable):
    """Docstring."""

    pass


class State(dict):
    """Docstring."""

    def split(self, *filters) -> tuple:
        """Docstring."""
        out = [State() for _ in filters]
        rest = State()
        for k, v in self.items():
            matched = False
            for i, f in enumerate(filters):
                if isinstance(v, f):
                    out[i][k] = v
                    matched = True
                    break
            if not matched:
                rest[k] = v
        return tuple(out) + (rest,)


def merge(*states: State) -> State:
    """Docstring."""
    res = State()
    for s in states:
        res.update(s)
    return res
