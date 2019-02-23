from ..utils.time_series import rand_intervals_fixed_n, rand_intervals_rand_n
import numpy as np

N_ITER = 10_000



def _test_rand_intervals(func):
    m = 100
    x = np.arange(m)
    for _ in range(N_ITER):
        intervals = func(x)
        assert np.issubdtype(intervals.dtype, np.integer)

        starts = intervals[:, 0]
        ends = intervals[:, 1]
        assert np.all(ends <= x.size)  # within bounds
        assert np.all(starts >= 0)  # within bounds
        assert np.all(ends > starts)  # only non-empty intervals


def _test_rand_intervals_random_state(func):
    m = 100
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
    _test_rand_intervals(rand_intervals_fixed_n)
    _test_rand_intervals_random_state(rand_intervals_fixed_n)

    # test number of intervals
    x = np.arange(10)
    for i in range(2, 100, 2):
        intervals = rand_intervals_fixed_n(x, n=i)
        assert intervals.shape[0] == i
