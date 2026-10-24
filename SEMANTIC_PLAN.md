# Semantic Implementation Plan

This document outlines the exhaustive checklist for implementing true mathematical and semantic behavior for the `zero-flax` API surface against `ml-ecosystem-snapshots` v0.0.4 (`flax_nnx_v0.12.10.json`).

Each checklist item requires:
1. Replacing any mock/stub implementation with actual mathematical logic using `ml-switcheroo-compiler` and `zero-jax` (`jnp`, `lax`, etc.).
2. Creating rigorous semantic unit tests that verify mathematical correctness (e.g., verifying forward pass values, edge cases, and gradient flow where applicable).

## JAX Core Activations & Operations (`jax.nn`)
- [x] `jax.nn.gelu` (Exact and approximate formulations)
- [x] `jax.nn.logsumexp` (Stable log-sum-exp reduction)
- [x] `jax.nn.one_hot` (Index encoding)
- [x] `jax.nn.softmax` (Stable softmax)

## JAX Initializers (`jax.nn.initializers`)
- [x] `constant`
- [x] `delta_orthogonal`
- [x] `glorot_normal` (Xavier normal)
- [x] `glorot_uniform` (Xavier uniform)
- [x] `he_normal` (Kaiming normal)
- [x] `he_uniform` (Kaiming uniform)
- [x] `kaiming_normal`
- [x] `kaiming_uniform`
- [x] `lecun_normal`
- [x] `lecun_uniform`
- [x] `normal`
- [x] `ones`
- [x] `orthogonal`
- [x] `truncated_normal`
- [x] `uniform`
- [x] `variance_scaling` (Core utility for adaptation)
- [x] `xavier_normal`
- [x] `xavier_uniform`
- [x] `zeros`

## Flax NNX Modules & Transformations (`flax.nnx`)
### Core Layers
- [x] `BatchNorm` (Running stats, momentum, spatial scaling/biasing)
- [x] `Bidirectional` (Forward and backward recurrent sequence wrapper)
- [x] `Conv` (n-D spatial convolutions, dilations, padding logic)
- [x] `ConvTranspose` (n-D transposed convolutions)
- [x] `Dense` (Standard dense linear layer)
- [x] `Dropout` (Stochastic masking during training, pass-through during inference)
- [x] `Einsum` (Learnable kernel/bias einsum operations)
- [x] `Embed` (Lookup tables for integer sequences)
- [x] `GRUCell` (Gated Recurrent Unit cell)
- [x] `GroupNorm` (Group-wise feature normalization)
- [x] `InstanceNorm` (Instance-wise spatial normalization)
- [x] `LSTMCell` (Long Short-Term Memory cell)
- [x] `LayerNorm` (Feature dimension normalization)
- [x] `Linear` (Dense matrix multiplications)
- [x] `LinearGeneral` (Flexible axis generalized dense mappings)
- [x] `LoRA` (Low-Rank Adaptation standalone layer)
- [x] `LoRALinear` (Linear layer combined with LoRA)
- [x] `MultiHeadAttention` (Query-Key-Value projection and scaled dot-product attention)
- [x] `MultiHeadDotProductAttention` (Multi-head dot-product attention alias/layer)
- [x] `OptimizedLSTMCell` (Fused-gate Long Short-Term Memory cell)
- [x] `PReLU` (Parametric Rectified Linear Unit)
- [x] `RMSNorm` (Root Mean Square normalization)
- [x] `RNN` (Recurrent cell sequence unrolling wrapper)
- [x] `RNNCellBase` (Abstract base class for recurrent cells)
- [x] `RoPE` (Rotary Positional Embedding)
- [x] `SimpleCell` (Elman / simple recurrent cell)
- [x] `SpectralNorm` (Power-iteration spectral normalization wrapper)
- [x] `WeightNorm` (Magnitude/direction weight normalization wrapper)

### Container Modules
- [x] `Dict`
- [x] `List`
- [x] `Sequential`

### Functional Transformations
- [x] `Jit`
- [x] `Pmap`
- [x] `Remat` (Gradient checkpointing)
- [x] `Scan` (Stateful looped unrolling)
- [x] `Vmap`

## Optax Losses (`optax.losses`)
- [x] `ctc_loss`
- [x] `ctc_loss_with_forward_probs`
- [x] `hinge_loss`
- [x] `huber_loss`
- [x] `l2_loss`
- [x] `make_fenchel_young_loss`
- [x] `multiclass_hinge_loss`
- [x] `multiclass_perceptron_loss`
- [x] `multiclass_sparsemax_loss`
- [x] `perceptron_loss`
- [x] `poly_loss_cross_entropy`
- [x] `ranking_softmax_loss`
- [x] `safe_softmax_cross_entropy`
- [x] `sigmoid_binary_cross_entropy`
- [x] `sigmoid_focal_loss`
- [x] `softmax_cross_entropy`
- [x] `softmax_cross_entropy_with_integer_labels`
- [x] `sparsemax_loss`
- [x] `squared_error`

## Optax Schedules (`optax.schedules`)
- [x] `constant_schedule`
- [x] `cosine_decay_schedule`
- [x] `cosine_onecycle_schedule`
- [x] `exponential_decay`
- [x] `inject_hyperparams`
- [x] `inject_stateful_hyperparams`
- [x] `join_schedules`
- [x] `linear_onecycle_schedule`
- [x] `linear_schedule`
- [x] `piecewise_constant_schedule`
- [x] `piecewise_interpolate_schedule`
- [x] `polynomial_schedule`
- [x] `sgdr_schedule`
- [x] `warmup_constant_schedule`
- [x] `warmup_cosine_decay_schedule`
- [x] `warmup_exponential_decay_schedule`
