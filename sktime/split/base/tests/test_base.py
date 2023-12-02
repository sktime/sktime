# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for base splitter class and functionality."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.tests._config import (
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    TEST_STEP_LENGTHS,
    TEST_WINDOW_LENGTHS,
)
from sktime.split.base._common import _inputs_are_supported
from sktime.split.expandingwindow import ExpandingWindowSplitter
from sktime.split.singlewindow import SingleWindowSplitter
from sktime.split.slidingwindow import SlidingWindowSplitter
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel
from sktime.utils._testing.series import _make_series

N_TIMEPOINTS = 30
TEST_Y_PANEL_HIERARCHICAL = [
    _make_hierarchical((2, 2), N_TIMEPOINTS, N_TIMEPOINTS),
    _make_panel(n_instances=2, n_timepoints=N_TIMEPOINTS),
]


def test_split_series():
    """Tests that split_series produces series in the split."""
    y = _make_series()
    cv = SlidingWindowSplitter()

    for train, test in cv.split_series(y):
        assert isinstance(train, pd.Series)
        assert len(train) == 10
        assert isinstance(test, pd.Series)
        assert len(test) == 1


def test_split_loc():
    """Tests that split_loc produces loc indices for train and test."""
    y = _make_series()
    cv = SlidingWindowSplitter()

    for train, test in cv.split_loc(y):
        assert isinstance(train, pd.DatetimeIndex)
        assert len(train) == 10
        y.loc[train]
        assert isinstance(test, pd.DatetimeIndex)
        assert len(test) == 1
        y.loc[test]


def test_split_series_hier():
    """Tests that split works with hierarchical data."""
    hierarchy_levels = (2, 4)
    n_instances = np.prod(hierarchy_levels)
    n = 12
    y = _make_hierarchical(
        hierarchy_levels=hierarchy_levels, max_timepoints=n, min_timepoints=n
    )
    cv = SlidingWindowSplitter()

    for train, test in cv.split(y):
        assert isinstance(train, np.ndarray)
        assert train.ndim == 1
        assert train.dtype == np.int64
        assert len(train) == 10 * n_instances
        assert isinstance(test, np.ndarray)
        assert test.ndim == 1
        assert pd.api.types.is_integer_dtype(test.dtype)
        assert len(test) == 1 * n_instances

    for train, test in cv.split_loc(y):
        assert isinstance(train, pd.MultiIndex)
        assert len(train) == 10 * n_instances
        assert train.isin(y.index).all()
        assert isinstance(test, pd.MultiIndex)
        assert len(test) == 1 * n_instances
        assert test.isin(y.index).all()

    def inst_index(y):
        return set(y.index.droplevel(-1).unique())

    for train, test in cv.split_series(y):
        assert isinstance(train, pd.DataFrame)
        assert len(train) == 10 * n_instances
        assert isinstance(test, pd.DataFrame)
        assert len(test) == 1 * n_instances
        assert inst_index(train) == inst_index(y)
        assert inst_index(test) == inst_index(y)


def test_hierarchical_singlewindowsplitter():
    """Test broadcasting of window splitters to hierarchical data.

    Also certifies for failure cas in bug #4972.
    """
    y = _make_hierarchical(hierarchy_levels=(2, 3), random_state=0)
    splitter = SingleWindowSplitter(fh=[1, 2], window_length=10)
    splits = list(splitter.split(y))
    assert len(splits) == 1, "Should only be one split"


@pytest.mark.parametrize("CV", [SlidingWindowSplitter, ExpandingWindowSplitter])
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_windowbase_splitter_get_n_split_hierarchical(
    CV, fh, window_length, step_length
):
    """Test that WindowBaseSplitter.get_n_splits works for hierarchical data."""
    # see bugs 4971
    y = TEST_Y_PANEL_HIERARCHICAL[0]  # hierarchical data
    if _inputs_are_supported([fh, window_length, step_length]):
        cv = CV(fh, window_length, step_length)
        assert cv.get_n_splits(y) == len(
            list(cv.split(y))
        ), "get_n_splits does not equal the number of splits in the output."


@pytest.mark.parametrize("y", TEST_Y_PANEL_HIERARCHICAL)
@pytest.mark.parametrize("CV", [SlidingWindowSplitter, ExpandingWindowSplitter])
def test_windowbase_splitter_get_n_split_unequal_series(y, CV):
    y_unequal = y.copy()  # avoid changing original dataset
    y_unequal.iloc[:3, :] = None  # make the first series shorter than the rest
    y_unequal.dropna(inplace=True)
    cv = CV([1], 24, 1)
    assert cv.get_n_splits(y_unequal) == len(
        list(cv.split(y_unequal))
    ), "get_n_splits does not equal the number of splits in the output."
