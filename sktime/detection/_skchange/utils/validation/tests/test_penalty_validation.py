import numpy as np
import pytest

from sktime.detection._skchange.utils.validation.penalties import (
    check_penalty,
    check_penalty_against_data,
)

"""Tests for penalty validation."""


@pytest.mark.parametrize(
    "penalty, arg_name, caller_name, require_constant_penalty, allow_none, allow_non_decreasing",  # noqa: E501
    [
        (None, "penalty", "test_func", False, True, False),
        (5.0, "penalty", "test_func", False, True, False),
        (np.array([1.0, 2.0, 3.0]), "penalty", "test_func", False, True, True),
        (np.array([1.0]), "penalty", "test_func", True, True, False),
    ],
)
def test_check_penalty_valid_cases(
    penalty,
    arg_name,
    caller_name,
    require_constant_penalty,
    allow_none,
    allow_non_decreasing,
):
    """Test valid cases for check_penalty."""
    assert (
        check_penalty(
            penalty,
            arg_name,
            caller_name,
            require_constant_penalty,
            allow_none,
            allow_non_decreasing,
        )
        is None
    )


@pytest.mark.parametrize(
    "penalty, arg_name, caller_name, require_constant_penalty, allow_none, allow_non_decreasing, expected_error",  # noqa: E501
    [
        (None, "penalty", "test_func", False, False, False, ValueError),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            "penalty",
            "test_func",
            False,
            True,
            False,
            ValueError,
        ),
        (np.array([]), "penalty", "test_func", False, True, False, ValueError),
        (np.array([1.0, 2.0]), "penalty", "test_func", True, True, False, ValueError),
        (-5.0, "penalty", "test_func", False, True, False, ValueError),
        (
            np.array([3.0, 2.0, 1.0]),
            "penalty",
            "test_func",
            False,
            True,
            False,
            ValueError,
        ),
        ("invalid", "penalty", "test_func", False, True, False, TypeError),
    ],
)
def test_check_penalty_invalid_cases(
    penalty,
    arg_name,
    caller_name,
    require_constant_penalty,
    allow_none,
    allow_non_decreasing,
    expected_error,
):
    """Test invalid cases for check_penalty."""
    with pytest.raises(expected_error):
        check_penalty(
            penalty,
            arg_name,
            caller_name,
            require_constant_penalty,
            allow_none,
            allow_non_decreasing,
        )


@pytest.mark.parametrize(
    "penalty, X, caller_name",
    [
        (5.0, np.random.rand(10, 3), "test_func"),
        (np.array([1.0, 2.0, 3.0]), np.random.rand(10, 3), "test_func"),
    ],
)
def test_check_penalty_against_data_valid_cases(penalty, X, caller_name):
    """Test valid cases for check_penalty_against_data."""
    assert check_penalty_against_data(penalty, X, caller_name) is None


@pytest.mark.parametrize(
    "penalty, X, caller_name, expected_error",
    [
        (np.array([1.0, 2.0]), np.random.rand(10, 3), "test_func", ValueError),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.random.rand(10, 3),
            "test_func",
            ValueError,
        ),
    ],
)
def test_check_penalty_against_data_invalid_cases(
    penalty, X, caller_name, expected_error
):
    """Test invalid cases for check_penalty_against_data."""
    with pytest.raises(expected_error):
        check_penalty_against_data(penalty, X, caller_name)
