# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-flax

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-flax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-flax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-79.9%25-yellow.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

## Why `zero-flax` Exists

This repository is a foundational component of the **Abstract ML Machine Ecosystem** (the `ml-switcheroo` ecosystem), designed to solve the **$N \times M$ translation problem** in Machine Learning. 

Currently, the ML landscape is heavily fragmented. If you write a model in JAX, PyTorch, Keras, or MLX (the $N$ frontends), deploying that model efficiently across WASM, WebGPU, TensorRT, or custom edge hardware (the $M$ backends) usually requires building and maintaining bespoke, complex translation pipelines for every single combination. 

### The Zero-Dependency Approach

`zero-flax` exists to address this by providing a **strictly zero external dependency** implementation of the Flax NNX API surface. It relies solely on the Python Standard Library, `numpy` (for eager mathematical evaluations), and `zero-jax`.

Instead of wrapping heavy C++ binaries or relying on XLA, `zero-flax` mimics the public Flax API—including neural network primitives like `Linear`, `Conv`, `MultiHeadAttention`, and `BatchNorm`, along with the functionalization of mutable state via the `flax.nnx` API standard—and acts as a pure Python frontend.

When you execute models using `zero-flax`, it routes operations through `zero-jax` which dynamically traces the operations using proxy tensors and delegates the logic to the `ml-switcheroo-compiler`. This compiler maps high-level API calls into a strictly defined Intermediate Representation (IR). The resulting IR can then be seamlessly consumed by various backends, enabling a robust **source-to-source** and **source-to-browser** compilation pipeline.

### Part of a Larger Ecosystem

`zero-flax` is not a standalone neural network library, but rather Tier 4 of the ML Switcheroo architecture:
1. **Tier 1 (`ml-switcheroo-ir`):** Defines the canonical logical graph dialect (ONNX spec compliance).
2. **Tier 2 (`ml-switcheroo-compiler`):** The computational heart, featuring AOT tracing, ProxyTensors, reverse-mode automatic differentiation, and optimizations like Dead Code Elimination (DCE).
3. **Tier 3 (`zero-jax`):** Provides the functional foundation and JAX API parity. Pytree flattening is used to safely route state into the compiler tape.
4. **Tier 4 (Frontends & Add-ons):** Repositories like `zero-flax`, `zero-optax`, and `zero-chex` build on top of `zero-jax` to provide Neural Network layers, optimizers, and typing without any heavy external dependencies.
5. **Tier 5 (`zero-zoo`):** Headless CI pipelines that train models deterministically to assert float-for-float equivalence ("Golden Seed" testing) across all simulated frameworks.

By perfectly mirroring the API signature of the authentic `flax.nnx` package (verified strictly via automated semantic snapshot testing, scoring against `flax_nnx_v0.8.5`), `zero-flax` allows users to drop it in as a lightweight substitute in environments where installing the massive official JAX/XLA stack is unfeasible—such as highly constrained serverless functions, or directly inside a web browser natively via Pyodide and PyScript.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
