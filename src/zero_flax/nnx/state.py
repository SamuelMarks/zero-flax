"""State and Variable tracking for nnx."""

from typing import Any


class Variable:
    """A base class representing a stateful variable in the nnx framework."""

    def __init__(self, value: Any) -> None:
        """Initializes the Variable with a value.

        Args:
            value: The initial value to store in the variable.
        """
        self.value = value

    def __repr__(self) -> str:
        """Returns a string representation of the Variable.

        Returns:
            A string showing the class name and the stored value.
        """
        return f"{self.__class__.__name__}({self.value})"


class Param(Variable):
    """A variable representing a trainable parameter."""

    pass


class BatchStat(Variable):
    """A variable representing a batch statistic, such as running mean or variance."""

    pass


class Rng(Variable):
    """A variable representing a random number generator state."""

    pass


class State(dict[str, Any]):
    """A dictionary-like container for storing a collection of Variables.

    This class extends the built-in dictionary to provide additional functionality
    for managing state, such as splitting the state based on variable types.
    """

    def split(self, *filters: Any) -> tuple[Any, ...]:
        """Splits the state into multiple states based on the provided filters.

        Each key-value pair in the state is evaluated against the filters in order.
        The value is placed into the first output state where it matches the filter type.
        Any remaining values that do not match any filter are placed in the last output state.

        Args:
            *filters: Variable types to filter the state by.

        Returns:
            A tuple containing a new State object for each filter, followed by an additional State object containing any unmatched values.
        """
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
    """Merges multiple State objects into a single State.

    If there are overlapping keys, the values from the later states in the arguments
    will overwrite the values from earlier states.

    Args:
        *states: The State objects to merge.

    Returns:
        A new State object containing the merged key-value pairs.
    """
    res = State()
    for s in states:
        res.update(s)
    return res
