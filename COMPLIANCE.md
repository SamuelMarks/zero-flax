# API Compliance Declaration

This framework is a frontend routing wrapper over the `ml-switcheroo-compiler` backend engine and `ml-switcheroo-ir` intermediate representation, validated against `ml-ecosystem-snapshots` v0.0.4 (`flax_nnx_v0.12.10.json`).
All mathematical and gradient computations evaluate through the central abstract IR.

- **100% Documentation Coverage**: Maintained across all files, modules, classes, functions, and arguments.
- **100% Test Coverage**: Maintained across all functions, lines, and branches.
- **Strong Typing**: Maintained with zero `Any`, zero `object`, zero `cast`, and zero reflection/dunder tricks.
- **No 3rd Party Executable Dependencies**: Maintained (defers strictly to `ml-switcheroo-ir`, `ml-switcheroo-compiler`, `ml-ecosystem-snapshots`, `pydantic`, `zero-*`, and standard library types).
