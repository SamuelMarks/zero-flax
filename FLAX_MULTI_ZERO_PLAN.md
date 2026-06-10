# `zero-flax` Migration to Native `zero-*` Ecosystem Dependencies

This document outlines the step-by-step checklist to replace the internal facade modules in `zero-flax` with their native `zero-*` ecosystem counterparts (`zero-chex`, `zero-optax`, `zero-orbax`, `zero-grain`, and `zero-jax`).

## Phase 1: Dependency Registration
- [ ] Update `pyproject.toml` to include the new ecosystem dependencies:
  - Add `"zero-chex"`
  - Add `"zero-optax"`
  - Add `"zero-orbax"`
  - Add `"zero-grain"`

## Phase 2: Purging Facade Modules
- [ ] Delete the `src/zero_flax/optax/` directory entirely.
- [ ] Delete the `src/zero_flax/jax/` directory entirely.
- [ ] Update `src/zero_flax/__init__.py` to remove exports for `jax` and `optax` (leaving only `__all__ = ["nnx"]`).

## Phase 3: Refactoring the `nnx` Subsystem
- [ ] In `src/zero_flax/nnx/missing.py`:
  - [ ] Delete the dummy `chex` class.
  - [ ] Delete the dummy `optax` class.
  - [ ] Delete the dummy `jax` class.
  - [ ] Add actual imports: `import zero_chex as chex`, `import zero_optax as optax`, `import zero_jax as jax`.
  - [ ] Fix any type hints in the missing module definitions to point to the real imported namespaces.
- [ ] In `src/zero_flax/nnx/__init__.py`:
  - [ ] Delete the dummy `chex` class.
  - [ ] Delete the dummy `optax` class.
  - [ ] Delete the dummy `jax` class.
  - [ ] Delete the local `initializers` class and `from . import initializers` import.
  - [ ] Add actual imports: `import zero_chex as chex`, `import zero_optax as optax`, `import zero_jax as jax`.
  - [ ] Import initializers directly from `zero_jax.nn`: `from zero_jax.nn import initializers`.
- [ ] Delete `src/zero_flax/nnx/initializers.py` if it is just a facade to `zero-jax`.

## Phase 4: Test Suite Cleanup
- [ ] Delete `tests/test_optax.py` (logic moved to `zero-optax`).
- [ ] Delete `tests/test_initializers.py` (logic moved to `zero-jax`).
- [ ] Review remaining tests (`test_e2e.py`, `test_layers.py`, etc.) and ensure any imports pointing to `zero_flax.jax` or `zero_flax.optax` are updated to point to `zero_jax` and `zero_optax`.

## Phase 5: Verification & Compliance
- [ ] Run `pytest` and ensure all tests pass.
- [ ] Run `ml_framework_snapshots check flax src/zero_flax --reference-prefix flax --target-prefix zero_flax` to ensure `flax.nnx` API parity remains at 100% without relying on the purged facades.
