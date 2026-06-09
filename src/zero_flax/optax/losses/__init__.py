from zero_optax.losses import ctc_loss
from zero_optax.losses import ctc_loss_with_forward_probs
from zero_optax.losses import hinge_loss
from zero_optax.losses import huber_loss
from zero_optax.losses import l2_loss
from zero_optax.losses import make_fenchel_young_loss
from zero_optax.losses import multiclass_hinge_loss
from zero_optax.losses import multiclass_perceptron_loss
from zero_optax.losses import multiclass_sparsemax_loss
from zero_optax.losses import perceptron_loss
from zero_optax.losses import poly_loss_cross_entropy
from zero_optax.losses import ranking_softmax_loss
from zero_optax.losses import safe_softmax_cross_entropy
from zero_optax.losses import sigmoid_binary_cross_entropy
from zero_optax.losses import sigmoid_focal_loss
from zero_optax.losses import softmax_cross_entropy
from zero_optax.losses import softmax_cross_entropy_with_integer_labels
from zero_optax.losses import sparsemax_loss
from zero_optax.losses import squared_error

__all__ = [
    "ctc_loss",
    "ctc_loss_with_forward_probs",
    "hinge_loss",
    "huber_loss",
    "l2_loss",
    "make_fenchel_young_loss",
    "multiclass_hinge_loss",
    "multiclass_perceptron_loss",
    "multiclass_sparsemax_loss",
    "perceptron_loss",
    "poly_loss_cross_entropy",
    "ranking_softmax_loss",
    "safe_softmax_cross_entropy",
    "sigmoid_binary_cross_entropy",
    "sigmoid_focal_loss",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
    "sparsemax_loss",
    "squared_error",
]
