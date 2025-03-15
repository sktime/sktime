# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for single window splitter."""

import numpy as np
import pytest

from sktime.forecasting.tests._config import (
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    TEST_WINDOW_LENGTHS,
    TEST_YS,
)
from sktime.split import SingleWindowSplitter
from sktime.split.base._common import _inputs_are_supported
from sktime.split.tests.test_split import _check_cv
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation import array_is_int
from sktime.utils.validation.forecasting import check_fh


@pytest.mark.skipif(
    not run_test_for_class(SingleWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_single_window_splitter(y, fh, window_length):
    """Test SingleWindowSplitter."""
    if _inputs_are_supported([fh, window_length]):
        cv = SingleWindowSplitter(fh=fh, window_length=window_length)
        train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)

        train_window = train_windows[0]
        test_window = test_windows[0]
        assert n_splits == 1
        assert train_window.shape[0] == _coerce_duration_to_int(
            duration=window_length, freq="D"
        )
        checked_fh = check_fh(fh)
        assert test_window.shape[0] == len(checked_fh)

        if array_is_int(checked_fh):
            test_window_expected = train_window[-1] + checked_fh
        else:
            test_window_expected = np.array(
                [y.index.get_loc(y.index[train_window[-1]] + x) for x in checked_fh]
            )
        np.testing.assert_array_equal(test_window, test_window_expected)
    else:
        with pytest.raises(TypeError, match="Unsupported combination of types"):
            SingleWindowSplitter(fh=fh, window_length=window_length)


@pytest.mark.skipif(
    not run_test_for_class(SingleWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
def test_single_window_splitter_default_window_length(y, fh):
    """Test SingleWindowSplitter."""
    cv = SingleWindowSplitter(fh=fh)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)

    train_window = train_windows[0]
    test_window = test_windows[0]

    assert n_splits == 1
    checked_fh = check_fh(fh)
    assert test_window.shape[0] == len(checked_fh)

    fh = cv.get_fh()
    if fh.is_all_in_sample():
        assert train_window.shape[0] == len(y)
    else:
        if array_is_int(checked_fh):
            assert train_window.shape[0] == len(y) - checked_fh.max()
        else:
            assert train_window.shape[0] == len(
                y[y.index <= y.index.max() - checked_fh.max()]
            )

    if array_is_int(checked_fh):
        test_window_expected = train_window[-1] + checked_fh
    else:
        test_window_expected = np.array(
            [y.index.get_loc(y.index[train_window[-1]] + x) for x in checked_fh]
        )
    np.testing.assert_array_equal(test_window, test_window_expected)
