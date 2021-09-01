# -*- coding: utf-8 -*-
import pytest

from sktime.utils.validation._dependencies import _check_soft_dependencies


def test_check_soft_dependencies_raises_error():
    """Test the _check_soft_dependencies() function."""
    with pytest.raises(ModuleNotFoundError, match=r".* soft dependency .*"):
        _check_soft_dependencies("unavailable_module")

    with pytest.raises(ModuleNotFoundError, match=r".* soft dependency .*"):
        _check_soft_dependencies("unavailable_module_1", "unavailable_module_2")
