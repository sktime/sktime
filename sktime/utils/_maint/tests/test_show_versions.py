"""Tests for the show_versions utility."""

import pathlib
import uuid

from sktime.utils._maint._show_versions import (
    DEFAULT_DEPS_TO_SHOW,
    _get_deps_info,
    _get_depstrs_from_estimator,
    _get_pkgnames_from_deptag,
    show_versions,
)
from sktime.utils.dependencies import _check_soft_dependencies


def test_show_versions_runs():
    """Test that show_versions runs without exceptions."""
    # only prints, should return None
    assert show_versions() is None


def test_deps_info():
    """Test that _get_deps_info returns package/version dict as per contract."""
    deps_info = _get_deps_info()
    assert isinstance(deps_info, dict)
    assert set(deps_info.keys()) == {"sktime"}

    deps_info_default = _get_deps_info(DEFAULT_DEPS_TO_SHOW)
    assert isinstance(deps_info_default, dict)
    assert set(deps_info_default.keys()) == set(DEFAULT_DEPS_TO_SHOW)

    KEY_ALIAS = {"sklearn": "scikit-learn", "skbase": "scikit-base"}

    for key in DEFAULT_DEPS_TO_SHOW:
        pkg_name = KEY_ALIAS.get(key, key)
        key_is_available = _check_soft_dependencies(pkg_name, severity="none")
        assert (deps_info_default[key] is None) != key_is_available
        if key_is_available:
            assert _check_soft_dependencies(f"{pkg_name}=={deps_info_default[key]}")
        deps_single_key = _get_deps_info([key])
        assert set(deps_single_key.keys()) == {key}


def test_deps_info_deps_missing_package_present_directory():
    """Test that _get_deps_info does not fail if a dependency is missing."""
    dummy_package_name = uuid.uuid4().hex

    dummy_folder_path = pathlib.Path(dummy_package_name)
    dummy_folder_path.mkdir()

    assert _get_deps_info([dummy_package_name]) == {dummy_package_name: None}

    dummy_folder_path.rmdir()


def test_deps_info_for_estimator_tag():
    """Test that _get_deps_info returns package/version dict as per contract."""
    deptag = [
        ("pandas>2.2.1", "numpy==1.23.4"),
        ["scikit-learn", ("matplotlib>=3.5", ["seaborn", "numpy<=1.24"])]
    ]
    pkgnames = _get_pkgnames_from_deptag(deptag)
    assert set(pkgnames) == {"pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"}


def test_get_depstrs_from_estimator():
    """Test that _get_depstrs_from_estimator returns package/version dict as per contract."""
    from sktime.forecasting.fbprophet import Prophet
    depstrs = _get_depstrs_from_estimator(Prophet)
    assert isinstance(depstrs, list)
    assert len(depstrs) > 0
    assert depstrs == ["prophet"]

def test_show_versions_for_estimator():
    """Test that show_versions runs without exceptions, with estimator arg."""
    from sktime.forecasting.fbprophet import Prophet
    # only prints, should return None
    assert show_versions(Prophet) is None
