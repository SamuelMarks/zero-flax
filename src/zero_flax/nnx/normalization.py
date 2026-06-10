from zero_flax.nnx.module import Module


class BatchNorm(Module):
    def __init__(self, features: int, *args, **kwargs):
        super().__init__()
        self.features = features
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        return x


class LayerNorm(Module):
    def __init__(self, features: int, *args, **kwargs):
        super().__init__()
        self.features = features
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        return x


class RMSNorm(Module):
    def __init__(self, features: int, *args, **kwargs):
        super().__init__()
        self.features = features
        self._is_initializing = False

    def __call__(self, x, *args, **kwargs):
        return x
