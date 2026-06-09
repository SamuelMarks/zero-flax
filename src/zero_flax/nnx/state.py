"""State and Variable tracking for nnx."""

from typing import Any  # noqa: F401


class Variable:
    """Base class for all state variables."""

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class Param(Variable):
    """Trainable parameters."""

    pass


class BatchStat(Variable):
    """Non-trainable batch statistics (e.g. BatchNorm running mean)."""

    pass


class Rng(Variable):
    """PRNG key state."""

    pass


class State(dict):
    """A dictionary-like container for Variables."""

    def split(self, *filters) -> tuple:
        """Splits state into multiple states based on Variable type filters."""
        # Simplified mock implementation
        results = [State() for _ in filters]
        remainder = State()

        for k, v in self.items():
            matched = False
            for i, f in enumerate(filters):
                if isinstance(f, type) and issubclass(f, Variable):
                    if isinstance(v, f) or (
                        isinstance(v, dict)
                        and any(isinstance(vv, f) for vv in v.values())
                    ):
                        results[i][k] = v
                        matched = True
                        break
            if not matched:
                remainder[k] = v

        return tuple(results) + (remainder,)


def merge(*states: State) -> State:
    """Merges multiple states into one."""
    merged = State()
    for state in states:
        merged.update(state)
    return merged
