from zero_flax.nnx.missing import (
    ConvTranspose,
    Jit,
    LoRA,
    LoRALinear,
    Pmap,
    Remat,
    Scan,
    Vmap,
)


def test_missing():
    # just instantiate
    ConvTranspose(2, 2, 1)()
    Jit(lambda x: x)()
    LoRA(2, 1, 3)()
    LoRALinear(2, 3, 1)()
    Pmap(lambda x: x)()
    Remat(lambda x: x)()
    Scan(lambda x: x)()
    Vmap(lambda x: x)()
