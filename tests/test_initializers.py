import numpy as np
from zero_flax.jax.nn import initializers


def test_zeros_ones():
    key = np.array([0, 0])
    z = initializers.zeros(key, (2, 2))
    assert np.all(z == 0)
    o = initializers.ones(key, (2, 2))
    assert np.all(o == 1)


def test_constant():
    init = initializers.constant(5)
    c = init(np.array([0, 0]), (2, 2))
    assert np.all(c == 5)


def test_distributions():
    key = np.array([0, 0])

    u = initializers.uniform(1.0)(key, (100, 100))
    assert np.mean(u) < 0.1

    n = initializers.normal(1.0)(key, (100, 100))
    assert np.mean(n) < 0.1

    tn = initializers.truncated_normal(1.0)(key, (100, 100))
    assert np.mean(tn) < 0.1


def test_variance_scaling():
    key = np.array([0, 0])
    shape = (10, 10)

    # Check that they run without error
    initializers.variance_scaling(1.0, "fan_in", "normal")(key, shape)
    initializers.glorot_uniform()(key, shape)
    initializers.glorot_normal()(key, shape)
    initializers.lecun_uniform()(key, shape)
    initializers.lecun_normal()(key, shape)
    initializers.he_uniform()(key, shape)
    initializers.he_normal()(key, shape)
    initializers.kaiming_uniform()(key, shape)
    initializers.kaiming_normal()(key, shape)
    initializers.xavier_uniform()(key, shape)
    initializers.xavier_normal()(key, shape)


def test_orthogonal():
    key = np.array([0, 0])
    # 2D
    o = initializers.orthogonal()(key, (5, 5))
    # Check orthogonality
    assert np.allclose(o @ o.T, np.eye(5), atol=1e-5)

    # Delta orthogonal
    do = initializers.delta_orthogonal()(key, (3, 3, 5, 5))
    assert do.shape == (3, 3, 5, 5)
