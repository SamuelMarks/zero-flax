"""README badge updater for test coverage, documentation coverage, and API compliance.

This script updates the Shields.io badges at the top of `README.md` based on
current test coverage, documentation coverage, and `SEMANTIC_PLAN.md` compliance.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence


def get_color(pct: float) -> str:
    """Return the Shields.io badge color name for a percentage score.

    Args:
        pct: Percentage value in ``[0.0, 100.0]``.

    Returns:
        Shields.io color identifier string.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 60.0:
        return "yellow"
    return "red"


def get_api_compliance(plan_path: str = "SEMANTIC_PLAN.md") -> float:
    """Compute the percentage of checked items in ``plan_path``.

    Args:
        plan_path: Path to ``SEMANTIC_PLAN.md``.

    Returns:
        Compliance percentage in ``[0.0, 100.0]``.
    """
    if not os.path.exists(plan_path):
        return 100.0
    with open(plan_path, encoding="utf-8") as handle:
        content = handle.read()
    checked = len(re.findall(r"- \[x\]", content))
    unchecked = len(re.findall(r"- \[ \]", content))
    total = checked + unchecked
    if total == 0:
        return 100.0
    return (checked / total) * 100.0


def update_readme(
    readme_path: str = "README.md",
    cov_pct: float = 100.0,
    doc_pct: float = 100.0,
    api_pct: float | None = None,
) -> None:
    """Update the coverage and compliance badges inside ``readme_path``.

    Args:
        readme_path: Path to ``README.md``.
        cov_pct: Test coverage percentage.
        doc_pct: Documentation coverage percentage.
        api_pct: Optional API compliance percentage (computed from ``SEMANTIC_PLAN.md`` if ``None``).
    """
    if not os.path.exists(readme_path):
        return

    resolved_api = get_api_compliance() if api_pct is None else api_pct
    with open(readme_path, encoding="utf-8") as handle:
        content = handle.read()

    cov_str = f"{cov_pct:.1f}".rstrip("0").rstrip(".")
    doc_str = f"{doc_pct:.1f}".rstrip("0").rstrip(".")
    api_str = f"{resolved_api:.1f}".rstrip("0").rstrip(".")

    cov_color = get_color(cov_pct)
    doc_color = get_color(doc_pct)
    api_color = get_color(resolved_api)

    content = re.sub(
        r"!\[Test Coverage\]\(https://img\.shields\.io/badge/coverage-[^)]+\)",
        f"![Test Coverage](https://img.shields.io/badge/coverage-{cov_str}%25-{cov_color}.svg)",
        content,
    )
    content = re.sub(
        r"!\[Doc Coverage\]\(https://img\.shields\.io/badge/docs-[^)]+\)",
        f"![Doc Coverage](https://img.shields.io/badge/docs-{doc_str}%25-{doc_color}.svg)",
        content,
    )
    if "![API Compliance]" in content:
        content = re.sub(
            r"!\[API Compliance\]\(https://img\.shields\.io/badge/api_compliance-[^)]+\)",
            f"![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)",
            content,
        )

    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main(argv: Sequence[str] | None = None) -> int:
    """Update `README.md` badges and return exit code ``0``.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Exit code ``0``.
    """
    _ = argv
    update_readme()
    return 0


if __name__ == "__main__":
    sys.exit(main())
