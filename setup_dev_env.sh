#!/bin/bash
set -e
echo "Setting up local editable environment..."
pip install "cdd @ git+https://github.com/offscale/cdd-python.git@master"
pip install --no-deps \
  "git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffelib" \
  "git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffecli" \
  "git+https://github.com/SamuelMarks/griffe.git@rest"
pip install --no-deps "https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl" || pip install --no-deps -e ../ml-ecosystem-snapshots
pip install "https://github.com/SamuelMarks/ml-switcheroo-ir/releases/download/v0.0.4/ml_switcheroo_ir-0.0.4-py3-none-any.whl" || pip install -e ../ml-switcheroo-ir
pip install "https://github.com/SamuelMarks/ml-switcheroo-compiler/releases/download/v0.0.1/ml_switcheroo_compiler-0.0.1-py3-none-any.whl" || pip install --no-deps -e ../ml-switcheroo-compiler
pip install --no-deps -e ../zero-jax
pip install --no-deps -e ../zero-chex
pip install --no-deps -e ../zero-grain
pip install --no-deps -e ../zero-optax
pip install --no-deps -e ../zero-orbax
pip install --no-deps -e .
echo "Done!"
