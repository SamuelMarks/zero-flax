# Semantic Implementation Plan

This document outlines the exhaustive checklist for implementing true mathematical and semantic behavior for the `zero-flax` API surface against `ml-ecosystem-snapshots` v0.0.5 (`flax_nnx_v0.12.10.json`).

Note: `grain`, `chex`, `orbax`, and `optax` are **not** replicated here; they are implemented in their dedicated sibling repositories (`../zero-grain`, `../zero-chex`, `../zero-orbax`, and `../zero-optax`).

Each checklist item requires:
1. Replacing any mock/stub implementation with actual mathematical logic using `ml-switcheroo-compiler` and `zero-jax` (`jnp`, `lax`, etc.).
2. Creating rigorous semantic unit tests that verify mathematical correctness (e.g., verifying forward pass values, edge cases, and gradient flow where applicable).

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
