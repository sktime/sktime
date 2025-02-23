__author__ = ["jgyasu"]

from unittest.mock import MagicMock

from sktime.utils.dependencies import _safe_import


def test_import_missing_module():
    """Test importing a module that is not installed."""
    result = _safe_import("nonexistent_module")
    assert isinstance(result, MagicMock)
    assert str(result) == (
        "Please install nonexistent_module to use this functionality."
    )


def test_import_without_pkg_name():
    """Test importing a module with the same name as package name."""
    result = _safe_import("torch")
    assert result is not None


def test_import_with_different_pkg_name_1():
    """Test importing a module with a different package name."""
    result = _safe_import("skbase.base.BaseObject", pkg_name="scikit-base")
    assert result is not None


def test_import_with_different_pkg_name_2():
    """Test importing another module with a different package name."""
    result = _safe_import("cv2", pkg_name="opencv-python")
    assert result is not None


def test_import_submodule():
    """Test importing a submodule."""
    result = _safe_import("torch.nn")
    assert result is not None


def test_import_class():
    """Test importing a class."""
    result = _safe_import("torch.nn.Linear")
    assert result is not None
