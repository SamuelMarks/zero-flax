"""Unit tests for all quality and compliance verification scripts in `scripts/`."""

from __future__ import annotations

from pathlib import Path

from scripts import (
    check_compiler_tests,
    check_compliance,
    check_deps,
    check_docs,
    check_parity_tests,
    lint_typing,
    update_badges,
    verify_parity,
)

NL = chr(10)


def test_lint_typing_script(tmp_path: Path) -> None:
    """Verify lint_typing passes on clean code and catches Any, object, cast, assert, reflection, and untyped defs.

    Args:
        tmp_path: Temporary pytest directory fixture.
    """
    assert lint_typing.main(["src/zero_flax", "scripts", "tests"]) == 0

    clean_file = tmp_path / "clean.py"
    clean_lines = [
        '"""Clean module."""',
        "",
        "def add(x: int, y: int) -> int:",
        '    """Add.',
        "",
        "    Args:",
        "        x: First.",
        "        y: Second.",
        "",
        "    Returns:",
        "        Sum.",
        '    """',
        "    return x + y",
    ]
    clean_file.write_text(NL.join(clean_lines) + NL, encoding="utf-8")
    assert lint_typing.main([str(clean_file)]) == 0

    src_dir = tmp_path / "src" / "zero_flax"
    src_dir.mkdir(parents=True)
    bad_file = src_dir / "bad.py"
    bad_lines = [
        "from typing import Any, cast",
        "# type:" + " ignore",
        "def bad(x, *args, **kwargs):",
        "    assert x is not None",
        "    y: Any = x.Any",
        "    z: object = cast(int, y)",
        "    w = int.cast(z)",
        "    getattr(w, 'a')",
        "    return w.__dict__",
    ]
    bad_file.write_text(NL.join(bad_lines) + NL, encoding="utf-8")
    assert lint_typing.main([str(src_dir)]) == 1


def test_check_docs_script(tmp_path: Path) -> None:
    """Verify check_docs passes on the repository and catches missing/stub docstrings and @typing.overload.

    Args:
        tmp_path: Temporary pytest directory fixture.
    """
    assert check_docs.main(["src/zero_flax", "scripts", "tests"]) == 0

    good_file = tmp_path / "good.py"
    good_lines = [
        '"""Module doc."""',
        "import typing",
        "from typing import overload",
        "@overload",
        "def f(x: int) -> int:",
        '    """Overload doc."""',
        "@typing.overload",
        "def f(x: float) -> float:",
        '    """Overload float doc."""',
        "def f(x: int | float) -> int | float:",
        '    """Compute f.',
        "",
        "    Args:",
        "        x: Input.",
        "",
        "    Returns:",
        "        Result.",
        '    """',
        "    return x",
    ]
    good_file.write_text(NL.join(good_lines) + NL, encoding="utf-8")
    assert check_docs.main([str(good_file)]) == 0

    bad_doc_file = tmp_path / "bad_doc.py"
    bad_doc_lines = [
        "class NoDoc:",
        "    def m1(self) -> None:",
        "        pass",
        "    def m2(self, a: int) -> int:",
        '        """Initialize."""',
        "        return a",
        "    def m3(self, a: int, *args: int, **kwargs: int) -> int:",
        '        """Missing args section."""',
        "        return a",
        "    def m4(self, a: int, b: int) -> int:",
        '        """Has Args:',
        "            a: Documented.",
        '        """',
        "        return a + b",
    ]
    bad_doc_file.write_text(NL.join(bad_doc_lines) + NL, encoding="utf-8")
    assert check_docs.main([str(tmp_path)]) == 1

    src_sub = tmp_path / "src" / "zero_flax"
    src_sub.mkdir(parents=True)
    bad_raise = src_sub / "bad_raise.py"
    bad_raise.write_text(
        NL.join(
            [
                '"""Module doc."""',
                "def r() -> None:",
                '    """Missing raises doc."""',
                "    raise ValueError('err')",
            ]
        )
        + NL,
        encoding="utf-8",
    )
    assert check_docs.main([str(bad_raise)]) == 1


def test_check_deps_script(tmp_path: Path) -> None:
    """Verify check_deps passes on allowed imports and rejects unauthorized 3rd-party packages.

    Args:
        tmp_path: Temporary pytest directory fixture.
    """
    assert check_deps.main(["src/zero_flax"]) == 0
    assert check_deps.is_stdlib("os")
    assert check_deps.is_stdlib("pathlib")
    assert check_deps.is_stdlib("sys")
    assert not check_deps.is_stdlib("pytest")
    assert not check_deps.is_stdlib("nonexistent_third_party_pkg_xyz")

    good_cross_imp = tmp_path / "good_cross_imp.py"
    good_cross_imp.write_text(
        NL.join(
            [
                "import zero_chex",
                "import zero_grain",
                "import zero_optax",
                "import zero_orbax",
                "import zero_jax",
                "from zero_chex import assert_shape",
                "from zero_optax import adamw",
                "from zero_orbax import checkpoint",
                "from zero_grain import DataLoader",
            ]
        ),
        encoding="utf-8",
    )
    assert check_deps.main([str(good_cross_imp)]) == 0

    bad_imp = tmp_path / "bad_imp.py"
    bad_imp.write_text(
        NL.join(["import nonexistent_third_party_pkg_xyz", "from another_bad_pkg import foo"]),
        encoding="utf-8",
    )
    assert check_deps.main([str(bad_imp)]) == 1

    syntax_err = tmp_path / "syntax_err.py"
    syntax_err.write_text("def broken(:" + NL, encoding="utf-8")
    assert check_deps.main([str(tmp_path)]) == 1


def test_check_compliance_script(tmp_path: Path) -> None:
    """Verify check_compliance against SEMANTIC_PLAN.md and synthetic plans.

    Args:
        tmp_path: Temporary pytest directory fixture.
    """
    assert check_compliance.main([]) == 0
    assert check_compliance._resolve_module_and_symbol("## Other", "Linear") == (
        "zero_flax.nnx",
        "Linear",
    )

    missing_plan = tmp_path / "MISSING.md"
    assert check_compliance.main([str(missing_plan)]) == 0

    bad_plan = tmp_path / "BAD_PLAN.md"
    bad_plan.write_text(
        NL.join(
            [
                "## Flax NNX (`flax.nnx`)",
                "- [x] NoBacktickLine",
                "- [x] `NonExistentLayerXYZ`",
                "## Bad Module (`optax.losses`)",
                "- [x] `NonExistentLossXYZ`",
            ]
        ),
        encoding="utf-8",
    )
    assert check_compliance.main([str(bad_plan)]) == 1


def test_check_compiler_and_parity_tests_scripts(tmp_path: Path) -> None:
    """Verify check_compiler_tests, check_parity_tests, and verify_parity scripts.

    Args:
        tmp_path: Temporary pytest directory fixture.
    """
    assert check_compiler_tests.main([]) == 0
    assert check_compiler_tests.main([str(tmp_path / "missing.py")]) == 1

    vacuous_parity = tmp_path / "vacuous.py"
    vacuous_parity.write_text(
        NL.join(
            [
                "def test_linear_parity() -> None:",
                "    pass",
                "def test_dense_parity() -> None:",
                "    x = print(1)",
                "    assert x.shape == (1, 1)",
                "def test_conv_parity() -> None:",
                "    assert_allclose(1, 1)",
            ]
        ),
        encoding="utf-8",
    )
    assert check_compiler_tests.main([str(vacuous_parity)]) == 1

    assert check_parity_tests.main([]) == 0
    assert check_parity_tests.verify_parity_references(str(tmp_path / "no_plan.md")) == []
    assert check_parity_tests.main(["SEMANTIC_PLAN.md", str(tmp_path / "no_test.py")]) == 1
    assert check_parity_tests.main(["SEMANTIC_PLAN.md", str(vacuous_parity)]) == 1

    assert verify_parity.main([]) == 0
    assert verify_parity.main([str(tmp_path / "missing.py")]) == 1
    assert verify_parity.main([str(vacuous_parity)]) == 1


def test_update_badges_script(tmp_path: Path) -> None:
    """Verify update_badges color mapping, compliance calculation, and README badge replacement.

    Args:
        tmp_path: Temporary pytest directory fixture.
    """
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(70.0) == "yellow"
    assert update_badges.get_color(40.0) == "red"

    assert update_badges.get_api_compliance(str(tmp_path / "missing.md")) == 100.0
    empty_plan = tmp_path / "empty.md"
    empty_plan.write_text("# Empty" + NL, encoding="utf-8")
    assert update_badges.get_api_compliance(str(empty_plan)) == 100.0

    half_plan = tmp_path / "half.md"
    half_plan.write_text(NL.join(["- [x] `A`", "- [ ] `B`"]), encoding="utf-8")
    assert update_badges.get_api_compliance(str(half_plan)) == 50.0

    readme = tmp_path / "README.md"
    readme.write_text(
        NL.join(
            [
                "# Title",
                "![Test Coverage](https://img.shields.io/badge/coverage-80%25-yellowgreen.svg)",
                "![Doc Coverage](https://img.shields.io/badge/docs-80%25-yellowgreen.svg)",
                "![API Compliance](https://img.shields.io/badge/api_compliance-80%25-yellowgreen.svg)",
            ]
        ),
        encoding="utf-8",
    )
    update_badges.update_readme(str(readme), cov_pct=100.0, doc_pct=100.0, api_pct=100.0)
    updated_text = readme.read_text(encoding="utf-8")
    assert "coverage-100%25-brightgreen.svg" in updated_text
    assert "docs-100%25-brightgreen.svg" in updated_text
    assert "api_compliance-100%25-brightgreen.svg" in updated_text

    readme_no_api = tmp_path / "README_NO_API.md"
    readme_no_api.write_text("# Title" + NL, encoding="utf-8")
    update_badges.update_readme(str(readme_no_api))

    update_badges.update_readme(str(tmp_path / "nonexistent_readme.md"))
    assert update_badges.main([]) == 0
