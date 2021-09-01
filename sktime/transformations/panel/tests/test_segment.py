# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from sktime.transformations.panel.segment import RandomIntervalSegmenter
from sktime.transformations.panel.segment import _rand_intervals_fixed_n
from sktime.transformations.panel.segment import _rand_intervals_rand_n
from sktime.utils._testing.panel import _make_nested_from_array

N_ITER = 10


# Test output format and dimensions.
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [0.1, 1.0, 1, 3, 10, "sqrt", "random", "log"])
def test_output_format_dim(n_timepoints, n_instances, n_intervals):
    X = _make_nested_from_array(
        np.ones(n_timepoints), n_instances=n_instances, n_columns=1
    )

    trans = RandomIntervalSegmenter(n_intervals=n_intervals)
    Xt = trans.fit_transform(X)

    # Check number of rows and output type.
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X.shape[0]

    # Check number of generated intervals/columns.
    if n_intervals != "random":
        if np.issubdtype(type(n_intervals), np.floating):
            assert Xt.shape[1] == np.maximum(1, int(n_timepoints * n_intervals))
        elif np.issubdtype(type(n_intervals), np.integer):
            assert Xt.shape[1] == n_intervals
        elif n_intervals == "sqrt":
            assert Xt.shape[1] == np.maximum(1, int(np.sqrt(n_timepoints)))
        elif n_intervals == "log":
            assert Xt.shape[1] == np.maximum(1, int(np.log(n_timepoints)))


# Check that exception is raised for bad input args.
@pytest.mark.parametrize("bad_interval", [0, -0, "str", 1.2, -1.2, -1])
def test_bad_input_args(bad_interval):
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=2)
    with pytest.raises(ValueError):
        RandomIntervalSegmenter(n_intervals=bad_interval).fit(X)


# Helper function for checking generated intervals.
@pytest.mark.parametrize(
    "random_state", list(np.random.randint(100, size=10))
)  # run repeatedly
@pytest.mark.parametrize("n_intervals", ["sqrt", "log", 0.1, 1, 3])
def test_rand_intervals_fixed_n(random_state, n_intervals):
    n_timepoints = 30
    x = np.arange(n_timepoints)

    intervals = _rand_intervals_fixed_n(x, n_intervals=n_intervals)
    assert intervals.ndim == 2
    assert np.issubdtype(intervals.dtype, np.integer)
    # assert intervals.shape[0] == np.unique(intervals, axis=0).shape[0]  #
    # no duplicates

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    assert np.all(ends <= n_timepoints)  # within bounds
    assert np.all(starts >= 0)  # within bounds
    assert np.all(ends > starts)  # only non-empty intervals


@pytest.mark.parametrize(
    "random_state", list(np.random.randint(100, size=10))
)  # run repeatedly
def test_rand_intervals_rand_n(random_state):
    n_timepoints = 30
    x = np.arange(n_timepoints)

    intervals = _rand_intervals_rand_n(x)
    assert intervals.ndim == 2
    assert np.issubdtype(intervals.dtype, np.integer)
    # assert intervals.shape[0] == np.unique(intervals, axis=0).shape[0]  #
    # no duplicates

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    assert np.all(ends <= x.size)  # within bounds
    assert np.all(starts >= 0)  # within bounds
    assert np.all(ends > starts)  # only non-empty intervals


@pytest.mark.parametrize("min_length", [1, 3])
@pytest.mark.parametrize("max_length", [4, 5])
@pytest.mark.parametrize("n_intervals", ["sqrt", "log", 0.1, 1, 3])
def test_rand_intervals_fixed_n_min_max_length(n_intervals, min_length, max_length):
    n_timepoints = 30
    x = np.arange(n_timepoints)

    intervals = _rand_intervals_fixed_n(
        x, n_intervals=n_intervals, min_length=min_length, max_length=max_length
    )
    lengths = intervals[:, 1] - intervals[:, 0]  # length = end - start
    assert np.all(lengths >= min_length)
    assert np.all(lengths <= max_length)
