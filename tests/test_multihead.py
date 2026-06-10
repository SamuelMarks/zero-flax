import pytest
import zero_flax.nnx as nnx


def test_multihead():
    m = nnx.MultiHeadAttention(2, 4)
    assert m(None, None, None) is None


def test_multihead2():
    pass


def test_multihead3():
    from zero_flax.nnx.layers import MultiHeadDotProductAttention

    m = MultiHeadDotProductAttention(2, 4)
    # The return None is at line 97, which might be skipped if we import from nnx and it's wrapped.
    # But it is returned.
    assert m(1, 2, 3) is None
