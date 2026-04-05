"""Regression tests for stream forecasting update wrappers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["SourishMerugu"]

import pytest

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.stream"),
    reason="test only if anything in sktime.forecasting.stream module has changed",
)
def test_update_refits_every_zero_lag_datetime_index():
    """Regression test: zero lag with datetime index should use all data.

    When refit_window_lag=0 (default) with a datetime-indexed series,
    get_window must return all available data, not just 1 point.

    Previously, _y.index[-0] was evaluated as _y.index[0] due to Python's
    -0 == 0 behaviour, causing get_window to return only 1 data point
    and the estimator to silently refit on a single observation.

    Regression test for the bug fixed in PR #9779.
    """
    from sktime.datasets import load_airline
    from sktime.forecasting.stream import UpdateRefitsEvery
    from sktime.forecasting.trend import TrendForecaster

    y = load_airline()
    f = UpdateRefitsEvery(TrendForecaster(), refit_interval=0, refit_window_lag=0)
    f.fit(y.iloc[:-20], fh=[1, 2, 3])
    f.update(y.iloc[-20:-10])
    assert f.cutoff[0] == y.index[-11]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.stream"),
    reason="test only if anything in sktime.forecasting.stream module has changed",
)
def test_update_every_zero_interval_datetime_index():
    """Regression test: update_interval=0 with datetime index should always update.

    Regression test for the bug fixed in PR #9779.
    """
    from sktime.datasets import load_airline
    from sktime.forecasting.stream import UpdateEvery
    from sktime.forecasting.trend import TrendForecaster

    y = load_airline()
    f = UpdateEvery(TrendForecaster(), update_interval=0)
    f.fit(y.iloc[:-20], fh=[1, 2, 3])
    f.update(y.iloc[-20:-10])
    assert f.cutoff[0] == y.index[-11]
