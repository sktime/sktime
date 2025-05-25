# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for sliding greedy splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.split import SlidingGreedySplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_lengths():
    """Test that SlidingGreedySplitter returns the correct lengths."""
    y = np.arange(10)
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=3)

    lengths = [(len(trn), len(tst)) for trn, tst in cv.split_series(y)]
    assert lengths == [(4, 2), (4, 2), (4, 2)]


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_forecast_horizon():
    """Test that SlidingGreedySplitter's forecast horizon is properly handled."""
    ts = np.arange(10)

    # Test with integer test_size
    cv_int_test = SlidingGreedySplitter(train_size=4, test_size=2, folds=3)
    assert len(cv_int_test.fh) == 2
    assert all(cv_int_test.fh == np.array([1, 2]))
    _ = list(cv_int_test.split(ts))  # Call to split should not affect fh initialization
    assert len(cv_int_test.fh) == 2

    # Test with float test_size
    cv_float_test = SlidingGreedySplitter(train_size=0.5, test_size=0.2, folds=2)
    assert cv_float_test.fh is None
    _ = list(cv_float_test.split(ts))  # Call split to should initialize fh
    assert len(cv_float_test.fh) == 2
    assert all(cv_float_test.fh == np.array([1, 2]))


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_indices():
    """Test that SlidingGreedySplitter returns the correct indices."""
    y = np.arange(10)
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=2)

    # Convert to list of tuples for easy comparison
    splits = [(list(trn), list(tst)) for trn, tst in cv.split(y)]

    # The expected indices for a sliding greedy splitter with train_size=4, test_size=2,
    # folds=2
    expected = [([2, 3, 4, 5], [6, 7]), ([4, 5, 6, 7], [8, 9])]

    # Compare actual and expected splits
    assert len(splits) == len(expected)
    for i, (actual, exp) in enumerate(zip(splits, expected)):
        assert actual[0] == exp[0], f"Train indices mismatch in fold {i}"
        assert actual[1] == exp[1], f"Test indices mismatch in fold {i}"


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_dates():
    """Test that SlidingGreedySplitter splits on dates correctly."""
    ts = _make_series(index_type="period")
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=3)

    train_starts = []
    train_ends = []
    test_starts = []
    test_ends = []

    for trn, tst in cv.split_series(ts):
        train_starts.append(trn.index[0])
        train_ends.append(trn.index[-1])
        test_starts.append(tst.index[0])
        test_ends.append(tst.index[-1])

    # Check that train/test windows slide properly
    assert test_ends[-1] == ts.index[-1]  # Last test ends at series end

    # Check correct number of folds
    assert len(train_starts) == 3
    assert len(test_starts) == 3

    # Check train/test sizes
    for trn, tst in cv.split_series(ts):
        assert len(trn) == 4  # train_size=4
        assert len(tst) == 2  # test_size=2

    # Check train/test contiguity and sliding behavior
    assert train_ends[0] == pd.Period("2003-08", "M")
    assert test_starts[0] == pd.Period("2003-09", "M")
    assert test_ends[0] == pd.Period("2003-10", "M")

    # Check that windows slide properly with correct step size
    assert train_starts[1] == pd.Period("2003-07", "M")  # Slides by test_size (2)
    assert test_starts[1] == pd.Period("2003-11", "M")

    # Check non-overlapping test windows
    assert test_ends[0] < test_starts[1]
    assert test_ends[1] < test_starts[2]

    # Check that test windows immediately follow train windows
    for i in range(len(train_ends)):
        next_month = train_ends[i].asfreq("M") + 1
        assert test_starts[i] == next_month

    # Check overall correctness of window positions
    expected_train_starts = [
        pd.Period("2003-05", "M"),
        pd.Period("2003-07", "M"),
        pd.Period("2003-09", "M"),
    ]
    expected_test_ends = [
        pd.Period("2003-10", "M"),
        pd.Period("2003-12", "M"),
        pd.Period("2004-02", "M"),
    ]

    assert train_starts == expected_train_starts
    assert test_ends == expected_test_ends


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_hierarchy():
    """Test that SlidingGreedySplitter handles uneven hierarchical data."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(2, 4),
        max_timepoints=20,
        min_timepoints=10,
        same_cutoff=False,
    )

    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=3)
    splits = list(cv.split_series(y_panel))

    # Test we have the expected number of folds
    assert len(splits) == 3

    for train, test in splits:
        # Check correct sizes for train and test sets
        assert all(len(group) == 4 for _, group in train.groupby(level=[0, 1]))
        assert all(len(group) == 2 for _, group in test.groupby(level=[0, 1]))

        # Verify hierarchical integrity is maintained
        assert set(train.index.droplevel(-1)) == set(y_panel.index.droplevel(-1))
        assert set(test.index.droplevel(-1)) == set(y_panel.index.droplevel(-1))

    # Test that the last test fold ends at the correct end date for each instance
    last_test_fold = splits[-1][1]

    last_test_dates_actual = (
        last_test_fold.reset_index(-1)
        .groupby(last_test_fold.index.names[:-1])
        .tail(1)["time"]
    )
    last_test_dates_expected = (
        y_panel.reset_index(-1).groupby(y_panel.index.names[:-1]).tail(1)["time"]
    )

    assert last_test_dates_actual.eq(last_test_dates_expected).all()

    # Verify that test sets are contiguous and follow train sets for each instance
    for i, (train, test) in enumerate(splits):
        # Group by hierarchy levels
        train_grouped = train.groupby(level=[0, 1])
        test_grouped = test.groupby(level=[0, 1])

        for group_key in train_grouped.groups:
            train_group = train_grouped.get_group(group_key)
            test_group = test_grouped.get_group(group_key)

            # Test set should immediately follow train set
            train_last_date = train_group.index.get_level_values("time").max()
            test_first_date = test_group.index.get_level_values("time").min()

            # The first date in test should be the next day after the last date in train
            next_day = pd.Timestamp(train_last_date) + pd.Timedelta(days=1)
            assert test_first_date == next_day


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_consecutive():
    """Test that SlidingGreedySplitter results in consecutive periods within windows."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(2, 4),
        max_timepoints=20,
        min_timepoints=10,
        same_cutoff=False,
    )

    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=3)

    for trn, tst in cv.split_series(y_panel):
        # Check train windows are consecutive
        train_consec = (
            trn.reset_index(-1)
            .groupby(trn.index.names[:-1])["time"]
            .agg(lambda s: len(s.diff().value_counts()) == 1)
        )
        assert train_consec.all()

        # Check test windows are consecutive
        test_consec = (
            tst.reset_index(-1)
            .groupby(tst.index.names[:-1])["time"]
            .agg(lambda s: len(s.diff().value_counts()) == 1)
        )
        assert test_consec.all()


@pytest.mark.skipif(
    not run_test_for_class(SlidingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_greedy_splitter_step_length():
    """Test that step_length parameter works correctly."""
    y = np.arange(15)
    cv = SlidingGreedySplitter(train_size=4, test_size=2, folds=3, step_length=3)

    splits = list(cv.split(y))

    # Check correct number of folds
    assert len(splits) == 3

    for train_idx, test_idx in splits:
        # Check train and test set sizes
        assert len(train_idx) == 4
        assert len(test_idx) == 2

    # For step_length=3, the test sets should be spaced 3 indices apart
    test_sets = [test for _, test in splits]

    # Check first fold's test set
    assert np.array_equal(test_sets[0], np.array([7, 8]))

    # Check second fold's test set (should be 3 steps before)
    assert np.array_equal(test_sets[1], np.array([10, 11]))

    # Check third fold's test set (should be 3 steps before)
    assert np.array_equal(test_sets[2], np.array([13, 14]))

    # Check that the train windows slide correctly
    train_sets = [train for train, _ in splits]
    assert np.array_equal(train_sets[0], np.array([3, 4, 5, 6]))
    assert np.array_equal(train_sets[1], np.array([6, 7, 8, 9]))
    assert np.array_equal(train_sets[2], np.array([9, 10, 11, 12]))

    # Confirm that step_length is respected between sets
    assert test_sets[1][0] - test_sets[0][0] == 3
    assert test_sets[2][0] - test_sets[1][0] == 3
