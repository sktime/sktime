# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for dependency checking utilities."""

import pytest

from sktime.utils.dependencies import _check_dl_dependencies, _check_mlflow_dependencies


def test_all_public_exports_importable():
    """Smoke test that all public exports from sktime.utils.dependencies import."""
    from sktime.utils.dependencies import (  # noqa: F401
        _check_dl_dependencies,
        _check_env_marker,
        _check_estimator_deps,
        _check_mlflow_dependencies,
        _check_python_version,
        _check_soft_dependencies,
        _isinstance_by_name,
        _placeholder_record,
        _safe_import,
    )


def test_check_dl_dependencies():
    """Test _check_dl_dependencies with severity='none'."""
    result = _check_dl_dependencies(severity="none")
    assert isinstance(result, bool)


def test_check_mlflow_dependencies():
    """Test _check_mlflow_dependencies with severity='none'."""

    result = _check_mlflow_dependencies(severity="none")
    assert isinstance(result, bool)


def test_check_dl_dependencies_error_when_missing():
    """Test _check_dl_dependencies raises error when tensorflow missing."""
    try:
        import tensorflow  # noqa: F401

        pytest.skip("tensorflow is installed")
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            _check_dl_dependencies(severity="error")


def test_check_mlflow_dependencies_error_when_missing():
    """Test _check_mlflow_dependencies raises error when mlflow missing."""
    try:
        import mlflow  # noqa: F401

        pytest.skip("mlflow is installed")
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            _check_mlflow_dependencies(severity="error")
