import numpy as np
import pytest

from sktime.detection._skchange.utils.validation.cuts import check_cuts_array


def test_check_cuts_array_valid():
    intervals = np.array([[1, 3], [4, 6], [7, 9]])
    result = check_cuts_array(intervals, n_samples=9)
    assert np.array_equal(result, intervals)


def test_check_cuts_array_invalid_ndim():
    intervals = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_cuts_array(intervals, n_samples=9)


def test_check_cuts_array_invalid_dtype():
    intervals = np.array([[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]])
    with pytest.raises(ValueError):
        check_cuts_array(intervals, n_samples=9)


def test_check_cuts_array_invalid_last_dim_size():
    intervals = np.array([[1, 3, 5], [4, 6, 8], [7, 9, 11]])
    with pytest.raises(ValueError):
        check_cuts_array(intervals, n_samples=11)


def test_check_cuts_array_not_strictly_increasing():
    intervals = np.array([[1, 3], [6, 4], [7, 9]])
    with pytest.raises(ValueError):
        check_cuts_array(intervals, n_samples=9)


def test_check_cuts_array_invalid_min_size():
    intervals = np.array([[1, 2], [4, 6], [7, 9]])
    with pytest.raises(ValueError):
        check_cuts_array(intervals, n_samples=9, min_size=3)


def test_check_cuts_array_negative_values():
    intervals = np.array([[1, 3], [-1, 6], [7, 9]])
    with pytest.raises(ValueError, match="All cuts must be non-negative"):
        check_cuts_array(intervals, n_samples=9)


def test_check_cuts_array_out_of_bounds():
    intervals = np.array([[1, 3], [4, 10], [7, 9]])
    with pytest.raises(ValueError, match="less than or equal to the number of samples"):
        check_cuts_array(intervals, n_samples=9)
