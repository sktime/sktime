# -*- coding: utf-8 -*-
"""Tests for the show_versions utility."""

from sktime.utils._maint._show_versions import (
    DEFAULT_DEPS_TO_SHOW,
    _get_deps_info,
    show_versions,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies


def test_show_versions_runs():
    """Test that show_versions runs without exeptions."""
    # only prints, should return None
    assert show_versions() is None


def test_deps_info():
    """Test that _get_deps_info returns package/version dict as per contract."""
    deps_info = _get_deps_info()
    assert isinstance(deps_info, dict)
    assert set(deps_info.keys()) == set(DEFAULT_DEPS_TO_SHOW)

    deps_info_with_set = _get_deps_info(DEFAULT_DEPS_TO_SHOW)
    assert isinstance(deps_info_with_set, dict)
    assert set(deps_info.keys()) == set(deps_info_with_set.keys())

    for key in DEFAULT_DEPS_TO_SHOW:
        assert deps_info[key] == _check_soft_dependencies(key, severity="none")
        assert deps_info[key] == deps_info_with_set[key]
        deps_single_key = _get_deps_info([key])
        assert set(deps_single_key.keys()) == {key}
        assert deps_info[key] == deps_single_key[key]