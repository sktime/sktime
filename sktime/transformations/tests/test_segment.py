"""Tests for RandomIntervalSegmenter."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.segment import (
    IntervalSegmenter,
    RandomIntervalSegmenter,
    _rand_intervals_fixed_n,
    _rand_intervals_rand_n,
)
from sktime.utils._testing.panel import _make_nested_from_array

N_ITER = 10


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_timepoints", [10, 20])
@pytest.mark.parametrize("n_intervals", [0.1, 1.0, 1, 3, 10, "sqrt", "random", "log"])
def test_output_format_dim(n_timepoints, n_instances, n_intervals):
    """Test output format and dimensions."""
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
            assert Xt.shape[1] <= np.maximum(1, int(n_timepoints * n_intervals))
        elif np.issubdtype(type(n_intervals), np.integer):
            assert Xt.shape[1] <= n_intervals
        elif n_intervals == "sqrt":
            assert Xt.shape[1] <= np.maximum(1, int(np.sqrt(n_timepoints)))
        elif n_intervals == "log":
            assert Xt.shape[1] <= np.maximum(1, int(np.log(n_timepoints)))


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bad_interval", [0, -0, "str", 1.2, -1.2, -1])
def test_bad_input_args(bad_interval):
    """Check that exception is raised for bad input args."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=2)
    with pytest.raises(ValueError):
        RandomIntervalSegmenter(n_intervals=bad_interval).fit(X)


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "random_state", list(np.random.randint(100, size=10))
)  # run repeatedly
@pytest.mark.parametrize("n_intervals", ["sqrt", "log", 0.1, 1, 3])
def test_rand_intervals_fixed_n(random_state, n_intervals):
    """Helper function for checking generated intervals."""
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


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "random_state", list(np.random.randint(100, size=10))
)  # run repeatedly
def test_rand_intervals_rand_n(random_state):
    """Test random intervals."""
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


@pytest.mark.skipif(
    not run_test_for_class(RandomIntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("min_length", [1, 3])
@pytest.mark.parametrize("max_length", [4, 5])
@pytest.mark.parametrize("n_intervals", ["sqrt", "log", 0.1, 1, 3])
def test_rand_intervals_fixed_n_min_max_length(n_intervals, min_length, max_length):
    """Test random interval length."""
    n_timepoints = 30
    x = np.arange(n_timepoints)

    intervals = _rand_intervals_fixed_n(
        x, n_intervals=n_intervals, min_length=min_length, max_length=max_length
    )
    lengths = intervals[:, 1] - intervals[:, 0]  # length = end - start
    assert np.all(lengths >= min_length)
    assert np.all(lengths <= max_length)


@pytest.mark.skipif(
    not run_test_for_class(IntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_interval_segmenter_integer_intervals_regression_11785():
    """Test IntervalSegmenter with integer intervals does not drop trailing time points.

    Regression test for #11785.
    """
    index = pd.MultiIndex.from_product([[0, 1], range(8)], names=["instance", "time"])
    X = pd.DataFrame(
        {
            "value": [
                5,
                7,
                6,
                8,
                9,
                4,
                3,
                5,
                1,
                2,
                2,
                3,
                4,
                4,
                5,
                6,
            ]
        },
        index=index,
    )

    t = IntervalSegmenter(intervals=2)
    Xt = t.fit_transform(X)

    # Expected exclusive end points [0, 4] and [4, 8]
    assert len(t.intervals_) == 2
    np.testing.assert_array_equal(t.intervals_[0], [0, 4])
    np.testing.assert_array_equal(t.intervals_[1], [4, 8])

    # Ensure all 4 timepoints per interval are retained for each instance
    for col in ["value_0_4", "value_4_8"]:
        assert col in Xt.columns
        for inst in [0, 1]:
            assert len(Xt.loc[inst, col]) == 4

    # Check exact values for instance 0
    np.testing.assert_array_equal(Xt.loc[0, "value_0_4"].values, [5, 7, 6, 8])
    np.testing.assert_array_equal(Xt.loc[0, "value_4_8"].values, [9, 4, 3, 5])

    # Check exact values for instance 1
    np.testing.assert_array_equal(Xt.loc[1, "value_0_4"].values, [1, 2, 2, 3])
    np.testing.assert_array_equal(Xt.loc[1, "value_4_8"].values, [4, 4, 5, 6])


@pytest.mark.skipif(
    not run_test_for_class(IntervalSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_timepoints", [10, 15, 20])
@pytest.mark.parametrize("n_intervals", [2, 3, 5])
def test_interval_segmenter_integer_intervals_partitions_series(
    n_timepoints, n_intervals
):
    """Test IntervalSegmenter partitions timepoints without gaps or drops."""
    index = pd.MultiIndex.from_product(
        [[0], range(n_timepoints)], names=["instance", "time"]
    )
    values = np.arange(n_timepoints)
    X = pd.DataFrame({"dim_0": values}, index=index)

    t = IntervalSegmenter(intervals=n_intervals)
    Xt = t.fit_transform(X)

    total_len = sum(len(Xt.loc[0, col].dropna()) for col in Xt.columns)
    assert total_len == n_timepoints

    concatenated = np.concatenate(
        [Xt.loc[0, col].dropna().values for col in Xt.columns]
    )
    np.testing.assert_array_equal(concatenated, values)
