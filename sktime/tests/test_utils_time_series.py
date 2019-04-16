from ..utils.time_series import rand_intervals_fixed_n, rand_intervals_rand_n, time_series_slope
from ..tests.test_RandomIntervalFeatureExtractor import generate_df_from_array
import numpy as np
from scipy.stats import linregress
import pytest

N_ITER = 100


def _test_rand_intervals(func, **kwargs):
    m = 30
    x = np.arange(m)
    for _ in range(N_ITER):
        intervals = func(x, **kwargs)
        assert intervals.ndim == 2
        assert np.issubdtype(intervals.dtype, np.integer)
        # assert intervals.shape[0] == np.unique(intervals, axis=0).shape[0]  # no duplicates

        starts = intervals[:, 0]
        ends = intervals[:, 1]
        assert np.all(ends <= x.size)  # within bounds
        assert np.all(starts >= 0)  # within bounds
        assert np.all(ends > starts)  # only non-empty intervals


def _test_rand_intervals_random_state(func):
    m = 10
    x = np.arange(m)
    random_state = 1234
    first_intervals = func(x, random_state=random_state)
    for _ in range(N_ITER):
        intervals = func(x, random_state=random_state)
        assert np.array_equal(first_intervals, intervals)


def test_rand_intervals_rand_n():
    _test_rand_intervals(rand_intervals_rand_n)
    _test_rand_intervals_random_state(rand_intervals_rand_n)


def test_rand_intervals_fixed_n():
    for n in [1, 3, 'sqrt']:
        _test_rand_intervals(rand_intervals_fixed_n, n=n)
    _test_rand_intervals_random_state(rand_intervals_fixed_n)

    # test number of intervals
    x = np.arange(10)
    for i in range(1, 100, 2):
        intervals = rand_intervals_fixed_n(x, n=i)
        assert intervals.shape[0] == i

    # test minimum length
    x = np.arange(200)
    for i in range(1, 20):
        intervals = rand_intervals_fixed_n(x, n=100, min_length=i)
        starts = intervals[:, 0]
        ends = intervals[:, 1]
        assert np.all(ends - starts >= i)  # minimum length


def test_bad_input_args():
    bad_n_intervals = [0, 'abc', 1.0]
    m = 10
    x = np.arange(m)
    for arg in bad_n_intervals:
        with pytest.raises(ValueError):
            rand_intervals_fixed_n(x, n=arg)


def test_time_series_slope():
    Y = np.array(generate_df_from_array(np.random.normal(size=10), n_rows=100).iloc[:, 0].tolist())
    y = Y[0, :]

    # Compare with scipy's linear regression function
    x = np.arange(y.size) + 1
    a = linregress(x, y).slope
    b = time_series_slope(y)
    np.testing.assert_almost_equal(a, b, decimal=10)

    # Check computations over axis
    a = np.apply_along_axis(time_series_slope, 1, Y)
    b = time_series_slope(Y, axis=1)
    np.testing.assert_equal(a, b)

    a = time_series_slope(Y, axis=1)[0]
    b = time_series_slope(y)
    np.testing.assert_equal(a, b)

    # Check linear and constant cases
    for step in [-1, 0, 1]:
        y = np.arange(1, 4) * step
        np.testing.assert_almost_equal(time_series_slope(y), step, decimal=10)




