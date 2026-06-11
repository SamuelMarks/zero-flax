# Refactoring Plan: Migrating `zero-flax` to the `zero-*` Ecosystem

## 1. Objective
Refactor `zero-flax` to depend heavily on its native sibling packages, eliminating stubs and mock implementations while adhering to strict quality and ecosystem purity constraints.

## 2. Core Quality Constraints
- **100% Documentation Coverage:** Every argument, function, class, file, and module must have a complete docstring.
- **100% Test Coverage:** All code must be tested, including 100% coverage for functions, lines, and branches.
- **Strong Typing:** The entire API surface and internal logic must use strict type hints, resolving all `Any` or missing types.
- **Zero 3rd-Party Dependencies:** No external third-party dependencies are allowed. The *only* permitted dependencies are:
  - `pydantic`
  - `../cdd-python`
  - `../ml-switcheroo-ir`
  - `../ml-switcheroo-compiler`
  - `../zero-jax`
  - `../zero-grain`
  - `../zero-orbax`
  - `../zero-chex`
  - `../zero-optax`
- **Pre-commit Integrity:** All pre-commit hooks (`pre-commit run --all-files`) must pass seamlessly without bypassing any checks.

---

## 3. Step-by-Step Execution Plan

### Phase 1: Dependency Overhaul
- [x] **Audit `pyproject.toml` and `requirements.txt`**:
  - Remove any implicit or explicit banned dependencies.
  - Introduce local directory references for the `zero-*` packages. Ensure the resolution paths look exactly like:
    - `zero-jax @ file://../zero-jax`
    - `zero-grain @ file://../zero-grain`
    - `zero-orbax @ file://../zero-orbax`
    - `zero-chex @ file://../zero-chex`
    - `zero-optax @ file://../zero-optax`
  - Verify that `ml-switcheroo-*`, `cdd-python`, and `pydantic` are properly declared if used.

### Phase 2: Facade & Stub Purging
- [x] **Delete Internal Mocks**: Remove all internal pseudo-modules and dummy facade classes currently substituting for the ecosystem packages.
  - Remove `src/zero_flax/jax/` (if present).
  - Remove `src/zero_flax/optax/` (if present).
  - Purge mock classes in `src/zero_flax/nnx/missing.py` and `__init__.py` that simulate `chex`, `jax`, and `optax`.

### Phase 3: Module Refactoring & Subsystem Wiring
- [x] **`zero-jax` Integration**:
  - Replace all array stubs with `zero_jax.numpy`.
  - Wire all initializers in `flax.nnx` layers to use `zero_jax.nn.initializers`.
  - Wire core math operations (e.g., convolution, matrix multiplication) to `zero_jax.lax`.
- [x] **`zero-chex` Integration**:
  - Implement `zero_chex` assertions inside modules to validate PyTree structures and array shapes dynamically.
  - Utilize `zero_chex` types for strict parameter and variable definitions.
- [x] **`zero-optax` Integration**:
  - Refactor training loop structures and state containers to utilize native `zero-optax` transformations and loss functions.
- [x] **`zero-orbax` Integration**:
  - Implement checkpointing utilities and state serialization processes leveraging `zero-orbax` interfaces.
- [x] **`zero-grain` Integration**:
  - Refactor data iterators and batch generation logic to depend natively on `zero-grain` hooks.

### Phase 4: Enforcing Quality Constraints
- [x] **Strong Typing Sweep**:
  - Run type checking (e.g., `mypy --strict`).
  - Refactor all `Any` annotations and ensure correct variance and generics on containers.
- [x] **Documentation Sweep**:
  - Audit every file in `src/zero_flax/`.
  - Ensure PEP-257 compliant docstrings exist for the module, every class, every function/method, and every parameter definition.
- [x] **Test Coverage Sweep**:
  - Refactor existing test suites (`test_e2e.py`, `test_layers.py`, etc.) to use the real wired-up `zero-*` ecosystem.
  - Execute `pytest --cov=src --cov-report=term-missing --cov-branch`.
  - Add granular edge-case tests to achieve exactly 100% line, function, and branch coverage.

### Phase 5: CI & Validation
- [x] **Pre-commit Execution**: Run `pre-commit run --all-files` and resolve any formatting (`ruff format`), linting (`ruff check`), or type errors.
- [x] **Final Build Validation**: Perform a local clean install (`pip install -e .`) and run the test suite to guarantee cross-dependency compatibility.
