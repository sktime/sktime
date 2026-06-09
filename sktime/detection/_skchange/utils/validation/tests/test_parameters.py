import pytest

from sktime.detection._skchange.utils.validation.parameters import check_smaller_than


def test_check_smaller_than_valid():
    assert check_smaller_than(10, 5, "test_param") == 5


def test_check_smaller_than_none_allowed():
    assert check_smaller_than(10, None, "test_param", allow_none=True) is None


def test_check_smaller_than_none_not_allowed():
    with pytest.raises(ValueError, match="test_param cannot be None."):
        check_smaller_than(10, None, "test_param")


def test_check_smaller_than_exceeds_max():
    with pytest.raises(ValueError, match="test_param must be at most 10"):
        check_smaller_than(10, 15, "test_param")
