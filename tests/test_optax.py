import numpy as np
from zero_flax.optax import losses, schedules


def test_losses():
    # just smoke test them since they are fully proxying well-tested optax in reality
    # for now they are mocked but let's test signature acceptance
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])

    losses.l2_loss(logits, labels)
    losses.huber_loss(logits, labels)
    losses.softmax_cross_entropy(logits, labels)


def test_schedules():
    sched = schedules.constant_schedule(0.1)
    assert sched(10) == 0.1

    sched2 = schedules.cosine_decay_schedule(0.1, 100)
    assert sched2(0) == 0.1
