# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Guard API reference autosummary entries against broken imports.

Regression guard for #4803: every ``autosummary`` entry under
``docs/source/api_reference`` must resolve to an importable object.
Skipped when ``docs/`` is not shipped (e.g. wheel installs).
"""

__author__ = ["shubham5080"]

import importlib
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_REF = _REPO_ROOT / "docs" / "source" / "api_reference"


def _iter_autosummary_entries(api_ref_dir):
    """Yield ``(rst_path, line_no, fully_qualified_name)`` from API rst files."""
    for rst in sorted(api_ref_dir.glob("*.rst")):
        current_module = None
        in_autosummary = False
        for line_no, line in enumerate(rst.read_text(encoding="utf-8").splitlines(), 1):
            current_match = re.match(r"\.\.\s+currentmodule::\s+(\S+)", line)
            if current_match:
                current_module = current_match.group(1)
                in_autosummary = False
                continue

            if re.match(r"\.\.\s+autosummary::", line):
                in_autosummary = True
                continue

            if not in_autosummary:
                continue

            if line and not line[0].isspace() and not line.startswith("\t"):
                in_autosummary = False
                continue

            stripped = line.strip()
            if not stripped or stripped.startswith(":") or stripped.startswith(".."):
                continue
            if current_module is None:
                continue

            yield rst, line_no, f"{current_module}.{stripped}"


def _resolve(qualname):
    """Import the longest possible module prefix, then ``getattr`` the rest."""
    parts = qualname.split(".")
    last_error = None
    for split in range(len(parts) - 1, 0, -1):
        modpath = ".".join(parts[:split])
        try:
            obj = importlib.import_module(modpath)
        except Exception as exc:  # noqa: BLE001 - collect and try shorter prefix
            last_error = exc
            continue
        try:
            for name in parts[split:]:
                obj = getattr(obj, name)
            return obj
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise ImportError(f"cannot resolve {qualname!r}") from last_error


@pytest.mark.skipif(
    not _API_REF.is_dir(),
    reason="docs/ not present in this install (e.g. wheel without docs)",
)
def test_api_reference_autosummary_entries_are_importable():
    """Every API reference autosummary entry must import successfully."""
    failures = []
    for rst, line_no, qualname in _iter_autosummary_entries(_API_REF):
        try:
            _resolve(qualname)
        except Exception as exc:  # noqa: BLE001 - report all failures together
            rel = rst.relative_to(_REPO_ROOT)
            failures.append(f"{rel}:{line_no}: {qualname} ({type(exc).__name__}: {exc})")

    assert not failures, (
        "Broken API reference autosummary imports "
        f"({len(failures)}):\n" + "\n".join(failures)
    )
