# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for sliding window splitter."""

import numpy as np
import pytest

from sktime.forecasting.tests._config import (
    TEST_FHS,
    TEST_FHS_TIMEDELTA,
    TEST_INITIAL_WINDOW,
    TEST_STEP_LENGTHS,
    TEST_WINDOW_LENGTHS,
    TEST_YS,
)
from sktime.split import SlidingWindowSplitter
from sktime.split.base._common import _inputs_are_supported
from sktime.split.tests.test_split import _check_cv, _get_n_incomplete_windows
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.series import _make_series
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation.forecasting import check_fh


@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_sliding_window_splitter(y, fh, window_length, step_length):
    """Test SlidingWindowSplitter."""
    if _inputs_are_supported([fh, window_length, step_length]):
        cv = SlidingWindowSplitter(
            fh=fh,
            window_length=window_length,
            step_length=step_length,
            start_with_window=True,
        )
        train_windows, test_windows, _, n_splits = _check_cv(cv, y)

        assert np.vstack(train_windows).shape == (
            n_splits,
            _coerce_duration_to_int(duration=window_length, freq="D"),
        )
        assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            SlidingWindowSplitter(
                fh=fh,
                window_length=window_length,
                step_length=step_length,
                start_with_window=True,
            )


@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
@pytest.mark.parametrize("initial_window", TEST_INITIAL_WINDOW)
def test_sliding_window_splitter_with_initial_window(
    y, fh, window_length, step_length, initial_window
):
    """Test SlidingWindowSplitter."""
    if _inputs_are_supported([fh, initial_window, window_length, step_length]):
        cv = SlidingWindowSplitter(
            fh=fh,
            window_length=window_length,
            step_length=step_length,
            initial_window=initial_window,
            start_with_window=True,
        )
        train_windows, test_windows, _, n_splits = _check_cv(cv, y)

        assert train_windows[0].shape[0] == _coerce_duration_to_int(
            duration=initial_window, freq="D"
        )
        assert np.vstack(train_windows[1:]).shape == (
            n_splits - 1,
            _coerce_duration_to_int(duration=window_length, freq="D"),
        )
        assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            SlidingWindowSplitter(
                fh=fh,
                initial_window=initial_window,
                window_length=window_length,
                step_length=step_length,
                start_with_window=True,
            )


@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", [*TEST_FHS, *TEST_FHS_TIMEDELTA])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_sliding_window_splitter_start_with_empty_window(
    y, fh, window_length, step_length
):
    """Test SlidingWindowSplitter."""
    if _inputs_are_supported([fh, window_length, step_length]):
        cv = SlidingWindowSplitter(
            fh=fh,
            window_length=window_length,
            step_length=step_length,
            start_with_window=False,
        )
        train_windows, test_windows, _, n_splits = _check_cv(
            cv, y, allow_empty_window=True
        )

        assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))

        n_incomplete = _get_n_incomplete_windows(window_length, step_length)
        train_windows = train_windows[n_incomplete:]

        assert np.vstack(train_windows).shape == (
            n_splits - n_incomplete,
            _coerce_duration_to_int(duration=window_length, freq="D"),
        )
    else:
        match = "Unsupported combination of types"
        with pytest.raises(TypeError, match=match):
            SlidingWindowSplitter(
                fh=fh,
                initial_window=None,
                window_length=window_length,
                step_length=step_length,
                start_with_window=False,
            )


@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_window_splitter_initial_window_start_with_empty_window_raises_error():
    """Test SlidingWindowSplitter."""
    y = _make_series()
    cv = SlidingWindowSplitter(
        fh=1,
        initial_window=15,
        start_with_window=False,
    )
    message = "`start_with_window` must be True if `initial_window` is given"
    with pytest.raises(ValueError, match=message):
        next(cv.split(y))


@pytest.mark.skipif(
    not run_test_for_class(SlidingWindowSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sliding_window_splitter_initial_window_smaller_than_window_raise_error():
    """Test SlidingWindowSplitter."""
    y = _make_series()
    cv = SlidingWindowSplitter(
        fh=1,
        window_length=10,
        initial_window=5,
    )
    message = "`initial_window` must greater than `window_length`"
    with pytest.raises(ValueError, match=message):
        next(cv.split(y))
