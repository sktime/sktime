# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Smoke tests for _safe_import (implementation lives in skbase)."""

from sktime.utils.dependencies import _safe_import


def test_safe_import_present():
    """Test that _safe_import returns a real module for installed package."""
    result = _safe_import("pandas")
    import pandas

    assert result is pandas


def test_safe_import_missing():
    """Test that _safe_import returns a mock for missing package."""
    result = _safe_import("nonexistent_module")
    assert result is not None
