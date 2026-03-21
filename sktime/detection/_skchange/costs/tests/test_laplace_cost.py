import numpy as np
import pytest

from sktime.detection._skchange.costs import LaplaceCost


def test_check_fixed_param_valid_parameters():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = (0.0, 1.0)
    result = cost._check_fixed_param(param, X)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.array_equal(result[0], np.array([0.0]))
    assert np.array_equal(result[1], np.array([1.0]))


def test_check_fixed_param_vector_parameters():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    result = cost._check_fixed_param(param, X)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.array_equal(result[0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(result[1], np.array([1.0, 2.0, 3.0]))


def test_check_fixed_param_not_tuple():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = [0.0, 1.0]  # List instead of tuple

    with pytest.raises(
        ValueError, match="Fixed Laplace parameters must be \\(location, scale\\)."
    ):
        cost._check_fixed_param(param, X)


def test_check_fixed_param_wrong_tuple_length():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = (0.0, 1.0, 2.0)  # Tuple with 3 elements

    with pytest.raises(
        ValueError, match="Fixed Laplace parameters must be \\(location, scale\\)."
    ):
        cost._check_fixed_param(param, X)


def test_check_fixed_param_invalid_scale():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = (0.0, -1.0)  # Negative scale

    with pytest.raises(ValueError, match="Parameter must be positive."):
        cost._check_fixed_param(param, X)


def test_check_fixed_param_invalid_scale_vector():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 3.0]))  # Zero scale

    with pytest.raises(ValueError, match="Parameter must be positive."):
        cost._check_fixed_param(param, X)


def test_check_fixed_param_wrong_dimensions():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    cost = LaplaceCost()
    param = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))  # Wrong dimensions

    with pytest.raises(ValueError, match="mean must have length 1 or X.shape\\[1\\]"):
        cost._check_fixed_param(param, X)
