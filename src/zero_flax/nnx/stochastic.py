from zero_flax.nnx.module import Module


class Dropout(Module):
    def __init__(self, rate: float, *args, **kwargs):
        super().__init__()
        self.rate = rate
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        return x
