# -*- coding: utf-8 -*-
import pytest

from sktime.utils.check_imports import _check_soft_deps


def test_check_soft_deps():
    """Test the _check_soft_deps() function."""
    with pytest.raises(ModuleNotFoundError, match=r".* soft dependency .*"):
        _check_soft_deps("unavailable_module")

    with pytest.raises(ModuleNotFoundError, match=r".* soft dependency .*"):
        _check_soft_deps("unavailable_module_1", "unavailable_module_2")
