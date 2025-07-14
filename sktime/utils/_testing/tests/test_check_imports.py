import pytest

from sktime.utils.dependencies import _check_soft_dependencies


def test_check_soft_dependencies_raises_error():
    """Test the _check_soft_dependencies() function."""
    MODULE = "unavailable_module"
    with pytest.raises(ModuleNotFoundError, match=r".* requires package .*"):
        _check_soft_dependencies(MODULE)

    with pytest.raises(ModuleNotFoundError, match=r".* requires package .*"):
        _check_soft_dependencies(MODULE + "_1", MODULE + "_2")
