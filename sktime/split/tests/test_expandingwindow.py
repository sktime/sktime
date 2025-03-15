# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for expanding window splitter."""

import numpy as np
import pytest

from sktime.forecasting.tests._config import (
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    TEST_STEP_LENGTHS,
    TEST_WINDOW_LENGTHS,
    TEST_YS,
)
from sktime.split import ExpandingWindowSplitter
from sktime.split.base._common import _inputs_are_supported
from sktime.split.tests.test_split import _check_cv, _get_n_incomplete_windows
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation.forecasting import check_fh


def _check_expanding_windows(windows):
    n_splits = len(windows)
    for i in range(1, n_splits):
        current = windows[i]
        previous = windows[i - 1]

        assert current.shape[0] > previous.shape[0]
        assert current[0] == previous[0]
        assert current[-1] > previous[-1]


@pytest.mark.skipif(
    not run_test_for_class(ExpandingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_expanding_window_splitter_start_with_initial_window_zero(y, fh, step_length):
    """Test ExpandingWindowSplitter."""
    initial_window = 0
    if _inputs_are_supported([fh, step_length, initial_window]):
        cv = ExpandingWindowSplitter(
            fh=fh,
            step_length=step_length,
            initial_window=initial_window,
        )
        train_windows, test_windows, _, n_splits = _check_cv(
            cv, y, allow_empty_window=True
        )

        assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            ExpandingWindowSplitter(
                fh=fh, initial_window=initial_window, step_length=step_length
            )


@pytest.mark.skipif(
    not run_test_for_class(ExpandingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("initial_window", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_expanding_window_splitter_start_with_empty_window(
    y, fh, initial_window, step_length
):
    """Test ExpandingWindowSplitter."""
    if _inputs_are_supported([fh, initial_window, step_length]):
        cv = ExpandingWindowSplitter(
            fh=fh,
            initial_window=initial_window,
            step_length=step_length,
        )
        train_windows, test_windows, _, n_splits = _check_cv(cv, y)
        assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))

        n_incomplete = _get_n_incomplete_windows(initial_window, step_length)
        train_windows = train_windows[n_incomplete:]
        _check_expanding_windows(train_windows)
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            ExpandingWindowSplitter(
                fh=fh,
                initial_window=initial_window,
                step_length=step_length,
            )


@pytest.mark.skipif(
    not run_test_for_class(ExpandingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("initial_window", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_expanding_window_splitter(y, fh, initial_window, step_length):
    """Test ExpandingWindowSplitter."""
    if _inputs_are_supported([fh, initial_window, step_length]):
        cv = ExpandingWindowSplitter(
            fh=fh,
            initial_window=initial_window,
            step_length=step_length,
        )
        train_windows, test_windows, _, n_splits = _check_cv(cv, y)
        assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))
        assert train_windows[0].shape[0] == _coerce_duration_to_int(
            duration=initial_window, freq="D"
        )
        _check_expanding_windows(train_windows)
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            ExpandingWindowSplitter(
                fh=fh,
                initial_window=initial_window,
                step_length=step_length,
            )
