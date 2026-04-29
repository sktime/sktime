"""Tests for stream forecasting update wrappers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["SourishMerugu"]

#import pytest

from sktime.forecasting.stream import DontUpdate, UpdateEvery, UpdateRefitsEvery
from sktime.forecasting.trend import TrendForecaster
#from sktime.utils.validation._dependencies import _check_soft_dependencies


def test_update_refits_every_zero_lag_datetime_index():
    """Regression test: zero lag with datetime index should use all data.

    When refit_window_lag=0 (default) with a datetime-indexed series,
    get_window must return all available data, not just 1 point.

    Previously, _y.index[-0] was evaluated as _y.index[0] due to Python's
    -0 == 0 behaviour, causing get_window to return only 1 data point.

    Regression test for bug fixed in PR #9779.
    """
    from sktime.datasets import load_airline

    y = load_airline()
    f = UpdateRefitsEvery(TrendForecaster(), refit_interval=0, refit_window_lag=0)
    f.fit(y.iloc[:-20], fh=[1, 2, 3])
    f.update(y.iloc[-20:-10])
    # cutoff should have moved forward by 10 steps
    assert f.cutoff[0] == y.index[-11]

def test_update_every_zero_interval_datetime_index():
    """Regression test: update_interval=0 with datetime index should always update.

    When update_interval=0, update should proceed without errors and
    the cutoff should advance correctly.

    Regression test for bug fixed in PR #9779.
    """
    from sktime.datasets import load_airline

    y = load_airline()
    f = UpdateEvery(TrendForecaster(), update_interval=0)
    f.fit(y.iloc[:-20], fh=[1, 2, 3])
    # should not raise, and cutoff should advance
    f.update(y.iloc[-20:-10])
    assert f.cutoff[0] == y.index[-11]

def test_update_refits_every_nonzero_lag():
    """Test that non-zero refit_window_lag correctly restricts the refit window."""
    from sktime.datasets import load_airline

    y = load_airline()
    f = UpdateRefitsEvery(TrendForecaster(), refit_interval=0, refit_window_lag=2)
    f.fit(y.iloc[:-20], fh=[1, 2, 3])
    f.update(y.iloc[-20:-10])
    assert f.cutoff[0] == y.index[-11]


def test_dont_update_never_changes_params():
    """Test that DontUpdate never updates model parameters after fit.

    Model parameters (slope, intercept) of the inner forecaster should
    remain unchanged after update, even though cutoff advances.
    """
    from sktime.datasets import load_airline

    y = load_airline()
    f = DontUpdate(TrendForecaster())
    f.fit(y.iloc[:-20], fh=[1, 2, 3])

    # capture inner model parameters before update
    slope_before = f.forecaster_.regressor_.coef_.copy()
    intercept_before = f.forecaster_.regressor_.intercept_.copy()

    f.update(y.iloc[-20:-10])

    # inner model parameters must be identical after update
    import numpy as np
    assert np.array_equal(slope_before, f.forecaster_.regressor_.coef_)
    assert np.array_equal(intercept_before, f.forecaster_.regressor_.intercept_)