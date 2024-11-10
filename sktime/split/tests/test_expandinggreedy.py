# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for expanding greedy splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.split import ExpandingGreedySplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class(ExpandingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expanding_greedy_splitter_lengths():
    """Test that ExpandingGreedySplitter returns the correct lengths."""
    y = np.arange(10)
    cv = ExpandingGreedySplitter(test_size=2, folds=3)

    lengths = [(len(trn), len(tst)) for trn, tst in cv.split_series(y)]
    assert lengths == [(4, 2), (6, 2), (8, 2)]


@pytest.mark.skipif(
    not run_test_for_class(ExpandingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expanding_greedy_splitter_dates():
    """Test that ExpandingGreedySplitter splits on dates correctly."""
    ts = _make_series(index_type="period")
    first_date = ts.index[0]
    last_date = ts.index[-1]
    cv = ExpandingGreedySplitter(test_size=2, folds=3)

    train_starts = []
    train_ends = []
    test_starts = []
    test_ends = []

    for trn, tst in cv.split_series(ts):
        train_starts.append(trn.index[0])
        train_ends.append(trn.index[-1])
        test_starts.append(tst.index[0])
        test_ends.append(tst.index[-1])

    assert train_starts == [first_date, first_date, first_date]
    assert train_ends == [last_date - 6, last_date - 4, last_date - 2]
    assert test_starts == [last_date - 5, last_date - 3, last_date - 1]
    assert test_ends == [last_date - 4, last_date - 2, last_date]


@pytest.mark.skipif(
    not run_test_for_class(ExpandingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expanding_greedy_splitter_hierarchy():
    """Test that ExpandingGreedySplitter handles uneven hierarchical data."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(2, 4),
        max_timepoints=20,
        min_timepoints=10,
        same_cutoff=False,
    )

    cv = ExpandingGreedySplitter(test_size=2, folds=3)

    # Since same_cutoff=False above, we will have a potentially different end date for
    # each instance in the hierarchy. Below we test that the final fold ends at the
    # correct end date for each instance
    last_test_fold = list(cv.split_series(y_panel))[-1][1]

    last_test_dates_actual = (
        last_test_fold.reset_index(-1)
        .groupby(last_test_fold.index.names[:-1])
        .tail(1)["time"]
    )
    last_test_dates_expected = (
        y_panel.reset_index(-1).groupby(y_panel.index.names[:-1]).tail(1)["time"]
    )

    assert last_test_dates_actual.eq(last_test_dates_expected).all()


@pytest.mark.skipif(
    not run_test_for_class(ExpandingGreedySplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_expanding_greedy_splitter_consecutive():
    """Test that ExpandingGreedySplitter results in consecutive test periods."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(2, 4),
        max_timepoints=20,
        min_timepoints=10,
        same_cutoff=False,
    )

    cv = ExpandingGreedySplitter(test_size=2, folds=3)

    test_dfs = [tst for _, tst in cv.split_series(y_panel)]

    combined_test_df = pd.concat(test_dfs).sort_index()

    # We check the .diff(). Equal diffs = uniformly spaced timestamps
    has_consecutive_index = (
        combined_test_df.reset_index(-1)
        .groupby(combined_test_df.index.names[:-1])["time"]
        .agg(lambda s: len(s.diff().value_counts()) == 1)
    )

    assert has_consecutive_index.all()
