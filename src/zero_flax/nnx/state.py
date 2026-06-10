"""State and Variable tracking for nnx."""

from typing import Any


class Variable:
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class Param(Variable):
    pass


class BatchStat(Variable):
    pass


class Rng(Variable):
    pass


class State(dict):
    def split(self, *filters) -> tuple:
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
    res = State()
    for s in states:
        res.update(s)
    return res
