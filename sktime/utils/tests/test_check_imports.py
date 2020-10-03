# -*- coding: utf-8 -*-
import pytest

from sktime.utils.check_imports import _check_imports


def test_check_imports():
    """Test the _check_imports() function."""
    with pytest.raises(Exception, match=r".* soft dependency .*"):
        _check_imports("unavailable_module")
