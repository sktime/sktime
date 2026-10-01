# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for cutoff splitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.tests._config import (
    TEST_CUTOFFS,
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    TEST_WINDOW_LENGTHS,
    TEST_YS,
)
from sktime.split import CutoffFhSplitter, CutoffSplitter
from sktime.split.base._common import _inputs_are_supported
from sktime.split.tests.test_split import _check_cv
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class([CutoffSplitter, ForecastingHorizon]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("cutoffs", TEST_CUTOFFS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_cutoff_window_splitter(y, cutoffs, fh, window_length):
    """Test CutoffSplitter."""
    if _inputs_are_supported([cutoffs, fh, window_length]):
        cv = CutoffSplitter(cutoffs, fh=fh, window_length=window_length)
        train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
        np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            CutoffSplitter(cutoffs, fh=fh, window_length=window_length)


@pytest.mark.skipif(
    not run_test_for_class([CutoffFhSplitter, ForecastingHorizon]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cutoff_fh_splitter():
    """Test CutoffFhSplitter."""
    from sktime.utils._testing.series import _make_series

    y = _make_series()
    cutoff = y.index[[10]]
    cutoff.freq = y.index.freq
    fh = ForecastingHorizon([1, 2, 3], freq=y.index.freq)

    spl = CutoffFhSplitter(cutoff, fh)

    spl_tt = list(spl.split_loc(y))[0]
    spl_train = spl_tt[0]
    spl_test = spl_tt[1]

    assert isinstance(spl_train, pd.DatetimeIndex)
    assert isinstance(spl_test, pd.DatetimeIndex)

    assert np.all(spl_train == y.index[:11])

    expected_test = pd.DatetimeIndex(
        ["2000-01-12", "2000-01-13", "2000-01-14"], dtype="datetime64[ns]", freq="D"
    )
    assert np.all(spl_test == expected_test)


@pytest.mark.skipif(
    not run_test_for_class([CutoffFhSplitter, ForecastingHorizon]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("fh", [None, [1, 2], ForecastingHorizon([1, 2])])
@pytest.mark.parametrize("method", ["split", "split_loc", "split_series"])
def test_cutoff_fh_splitter_relative_cutoffs(hierarchical, fh, method):
    """Test relative cutoffs and horizons on single and hierarchical series."""
    index = pd.date_range("2000-01-01", periods=8, freq="D")
    if hierarchical:
        index = pd.MultiIndex.from_product([["a", "b"], index])
    y = pd.DataFrame(np.arange(len(index)), index=index)
    spl = CutoffFhSplitter([3, 4], fh)

    splits = list(getattr(spl, method)(y))

    assert len(splits) == 2
    for cutoff, (train, test) in zip([3, 4], splits):
        positions = np.arange(len(y)) % 8
        expected_train = np.flatnonzero(positions <= cutoff)
        end = 7 if fh is None else cutoff + 2
        expected_test = np.flatnonzero((positions > cutoff) & (positions <= end))
        if method == "split":
            np.testing.assert_array_equal(train, expected_train)
            np.testing.assert_array_equal(test, expected_test)
        elif method == "split_loc":
            pd.testing.assert_index_equal(train, y.index[expected_train])
            pd.testing.assert_index_equal(test, y.index[expected_test])
        else:
            pd.testing.assert_frame_equal(train, y.iloc[expected_train])
            pd.testing.assert_frame_equal(test, y.iloc[expected_test])
