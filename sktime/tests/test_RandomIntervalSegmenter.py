from ..transformers.series_to_tabular import RandomIntervalSegmenter
from ..utils.testing import generate_df_from_array
from ..utils.transformations import tabularize
import pytest
import pandas as pd
import numpy as np

N_ITER = 10


# Test output format and dimensions.
def test_output_format_dim():
    for n_cols in [1, 3]:
        for n_rows in [1, 3]:
            for n_obs in [2, 100]:
                for n_intervals in [0.1, 0.5, 1.0, 1, 3, 10, 'sqrt', 'random', 'log']:
                    X = generate_df_from_array(np.ones(n_obs), n_rows=n_rows, n_cols=n_cols)

                    trans = RandomIntervalSegmenter(n_intervals=n_intervals)
                    Xt = trans.fit_transform(X)

                    # Check number of rows and output type.
                    assert isinstance(Xt, (pd.DataFrame, pd.Series))
                    assert Xt.shape[0] == X.shape[0]

                    # Check number of generated intervals/columns.
                    if n_intervals != 'random':
                        if np.issubdtype(type(n_intervals), np.float):
                            assert Xt.shape[1] == np.maximum(1, int(n_obs * n_intervals)) * n_cols
                        elif np.issubdtype(type(n_intervals), np.integer):
                            assert Xt.shape[1] == n_intervals * n_cols
                        elif n_intervals == 'sqrt':
                            assert Xt.shape[1] == np.maximum(1, int(np.sqrt(n_obs))) * n_cols
                        elif n_intervals == 'log':
                            assert Xt.shape[1] == np.maximum(1, int(np.log(n_obs))) * n_cols


# Check that exception is raised for bad input args.
def test_bad_input_args():
    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=2)
    bad_n_intervals = [0, -0, 'str', 1.2, -1.2, -1]
    for arg in bad_n_intervals:
        with pytest.raises(ValueError):
            RandomIntervalSegmenter(n_intervals=arg).fit(X)


# Check that random state always gives same result.
def test_random_state():
    X = generate_df_from_array(np.random.normal(size=10))
    random_state = 1234

    for n_intervals in [0.5, 10, 'sqrt', 'random', 'log']:
        trans = RandomIntervalSegmenter(n_intervals=n_intervals, random_state=random_state)
        first_Xt = trans.fit_transform(X)
        for _ in range(N_ITER):
            trans = RandomIntervalSegmenter(n_intervals=n_intervals, random_state=random_state)
            Xt = trans.fit_transform(X)
            np.testing.assert_array_equal(tabularize(first_Xt).values, tabularize(Xt).values)


# Helper function for checking generated intervals.
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


# Check intervals for random n.
def test_rand_intervals_rand_n():
    tran = RandomIntervalSegmenter()
    func = tran._rand_intervals_rand_n
    _test_rand_intervals(func)


# Check intervals for fixed n.
def test_rand_intervals_fixed_n():
    tran = RandomIntervalSegmenter()
    func = tran._rand_intervals_fixed_n

    for n in [0.5, 1, 3, 'sqrt', 'log']:
        _test_rand_intervals(func, n=n)


# Check minimum length.
def test_min_length():
    x = np.arange(200)
    for min_length in range(1, 20):
        for n_intervals in [0.5, 10, 'sqrt', 'log']:
            tran = RandomIntervalSegmenter(n_intervals=n_intervals, min_length=min_length)
            intervals = tran._rand_intervals_fixed_n(x, n=n_intervals)
            starts = intervals[:, 0]
            ends = intervals[:, 1]
            assert np.all(ends - starts >= min_length)  # minimum length
