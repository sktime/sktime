import numpy as np
import pandas as pd
import pytest
from sktime.transformers.series_as_features.segment import \
    RandomIntervalSegmenter
from sktime.utils.data_container import tabularize
from sktime.utils._testing import generate_df_from_array

N_ITER = 10


# Test output format and dimensions.
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals",
                         [0.1, 1.0, 1, 3, 10, 'sqrt', 'random', 'log'])
def test_output_format_dim(n_timepoints, n_instances, n_intervals):
    X = generate_df_from_array(np.ones(n_timepoints), n_rows=n_instances,
                               n_cols=1)

    trans = RandomIntervalSegmenter(n_intervals=n_intervals)
    Xt = trans.fit_transform(X)

    # Check number of rows and output type.
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X.shape[0]

    # Check number of generated intervals/columns.
    if n_intervals != 'random':
        if np.issubdtype(type(n_intervals), np.floating):
            assert Xt.shape[1] == np.maximum(1,
                                             int(n_timepoints * n_intervals))
        elif np.issubdtype(type(n_intervals), np.integer):
            assert Xt.shape[1] == n_intervals
        elif n_intervals == 'sqrt':
            assert Xt.shape[1] == np.maximum(1, int(np.sqrt(n_timepoints)))
        elif n_intervals == 'log':
            assert Xt.shape[1] == np.maximum(1, int(np.log(n_timepoints)))


# Check that exception is raised for bad input args.
@pytest.mark.parametrize("bad_interval", [0, -0, 'str', 1.2, -1.2, -1])
def test_bad_input_args(bad_interval):
    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=2)
    with pytest.raises(ValueError):
        RandomIntervalSegmenter(n_intervals=bad_interval).fit(X)


# Check that random state always gives same result.
def test_random_state():
    X = generate_df_from_array(np.random.normal(size=10))
    random_state = 1234

    for n_intervals in [0.5, 10, 'sqrt', 'random', 'log']:
        trans = RandomIntervalSegmenter(n_intervals=n_intervals,
                                        random_state=random_state)
        first_Xt = trans.fit_transform(X)
        for _ in range(N_ITER):
            trans = RandomIntervalSegmenter(n_intervals=n_intervals,
                                            random_state=random_state)
            Xt = trans.fit_transform(X)
            np.testing.assert_array_equal(tabularize(first_Xt).values,
                                          tabularize(Xt).values)


# Helper function for checking generated intervals.
@pytest.mark.parametrize("random_state", list(
    np.random.randint(100, size=10)))  # run repeatedly
@pytest.mark.parametrize("n_intervals", ['sqrt', 'log', 0.1, 1, 3])
def test_rand_intervals_fixed_n(random_state, n_intervals):
    tran = RandomIntervalSegmenter(random_state=random_state)
    series_len = 30
    x = np.arange(series_len)

    intervals = tran._rand_intervals_fixed_n(x, n_intervals=n_intervals)
    assert intervals.ndim == 2
    assert np.issubdtype(intervals.dtype, np.integer)
    # assert intervals.shape[0] == np.unique(intervals, axis=0).shape[0]  #
    # no duplicates

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    assert np.all(ends <= x.size)  # within bounds
    assert np.all(starts >= 0)  # within bounds
    assert np.all(ends > starts)  # only non-empty intervals


@pytest.mark.parametrize("random_state", list(
    np.random.randint(100, size=10)))  # run repeatedly
def test_rand_intervals_rand_n(random_state):
    tran = RandomIntervalSegmenter(random_state=random_state)
    series_len = 30
    x = np.arange(series_len)

    intervals = tran._rand_intervals_rand_n(x)
    assert intervals.ndim == 2
    assert np.issubdtype(intervals.dtype, np.integer)
    # assert intervals.shape[0] == np.unique(intervals, axis=0).shape[0]  #
    # no duplicates

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    assert np.all(ends <= x.size)  # within bounds
    assert np.all(starts >= 0)  # within bounds
    assert np.all(ends > starts)  # only non-empty intervals


# Check minimum length.
@pytest.mark.parametrize("min_length", [1, 3])
@pytest.mark.parametrize("n_intervals", ['sqrt', 'log', 0.1, 1, 3])
def test_min_length(n_intervals, min_length):
    series_len = 30
    x = np.arange(series_len)

    tran = RandomIntervalSegmenter(n_intervals=n_intervals,
                                   min_length=min_length)
    intervals = tran._rand_intervals_fixed_n(x, n_intervals=n_intervals)
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    assert np.all(ends - starts >= min_length)  # minimum length
