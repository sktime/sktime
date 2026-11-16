"""Tests for detector update compositors."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from sktime.detection.compose import RefitDetector, UpdateRefitsEvery
from sktime.detection.dummy import DummyRegularAnomalies


class _CounterDetector(DummyRegularAnomalies):
    """Spy detector that counts fit and update invocations."""

    def __init__(self):
        super().__init__()
        self.fit_count = 0
        self.update_count = 0
        self.last_fit_len = 0

    def _fit(self, X, y=None):
        self.fit_count += 1
        self.last_fit_len = len(X)
        return super()._fit(X, y)

    def _update(self, X, y=None):
        self.update_count += 1
        return self


def test_update_refits_every_basic():
    """Test basic fit, update and predict with UpdateRefitsEvery."""
    X = pd.DataFrame({"a": np.arange(30, dtype=float)})
    X0 = X.iloc[:10]
    X1 = X.iloc[10:20]
    X2 = X.iloc[20:]

    inner = DummyRegularAnomalies()
    wrapped = UpdateRefitsEvery(detector=inner, refit_interval=5)

    wrapped.fit(X0)
    assert wrapped.is_fitted
    assert wrapped.detector_.is_fitted

    wrapped.update(X1)
    assert len(wrapped._X) == 20

    y_pred = wrapped.predict(X2)
    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) <= len(X2)


def test_refit_detector_basic():
    """Test RefitDetector shorthand."""
    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    X0 = X.iloc[:10]
    X1 = X.iloc[10:]

    wrapped = RefitDetector(DummyRegularAnomalies())
    wrapped.fit(X0)
    assert wrapped.is_fitted

    wrapped.update(X1)
    assert len(wrapped._X) == 20

    y_pred = wrapped.predict(X1)
    assert isinstance(y_pred, pd.DataFrame)


@pytest.mark.parametrize("refit_interval", [0, 5])
def test_update_refits_every_window_slicing(refit_interval):
    """Test UpdateRefitsEvery with refit_window_size."""
    X = pd.DataFrame({"a": np.arange(30, dtype=float)})
    X0 = X.iloc[:10]
    X1 = X.iloc[10:25]

    wrapped = UpdateRefitsEvery(
        DummyRegularAnomalies(),
        refit_interval=refit_interval,
        refit_window_size=10,
    )
    wrapped.fit(X0)
    wrapped.update(X1)
    assert wrapped.is_fitted


def test_update_refits_every_spy_counter_range_index():
    """Verify fit() is called only when delta >= refit_interval on RangeIndex."""
    X = pd.DataFrame({"a": np.arange(30, dtype=float)})
    X0 = X.iloc[:10]
    X1 = X.iloc[10:14]  # delta = 13 - 9 = 4 (< 5)
    X2 = X.iloc[14:20]  # total len 20, delta = 19 - 9 = 10 (>= 5)

    spy = _CounterDetector()
    wrapped = UpdateRefitsEvery(detector=spy, refit_interval=5)

    wrapped.fit(X0)
    assert wrapped.detector_.fit_count == 1
    assert wrapped.detector_.last_fit_len == 10

    # delta = 4 < refit_interval 5 -> update called, fit NOT called
    wrapped.update(X1)
    assert wrapped.detector_.fit_count == 1
    assert wrapped.detector_.update_count == 1

    # delta = 10 >= refit_interval 5 -> refits on all 20 seen observations
    wrapped.update(X2)
    assert wrapped.detector_.fit_count == 2
    assert wrapped.detector_.last_fit_len == 20


def test_update_refits_every_spy_counter_datetime_index():
    """Verify fit() is called only when delta >= refit_interval on DatetimeIndex."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    X = pd.DataFrame({"a": np.arange(30, dtype=float)}, index=dates)
    X0 = X.iloc[:10]  # Jan 1 to Jan 10
    X1 = X.iloc[10:13]  # Jan 11 to Jan 13, delta = 3 days (< 5 days)
    X2 = X.iloc[13:20]  # Jan 14 to Jan 20, total len 20, delta = 10 days (>= 5 days)

    spy = _CounterDetector()
    wrapped = UpdateRefitsEvery(detector=spy, refit_interval=pd.Timedelta(days=5))

    wrapped.fit(X0)
    assert wrapped.detector_.fit_count == 1
    assert wrapped.detector_.last_fit_len == 10

    # delta = 3 days < 5 days -> update called, fit NOT called
    wrapped.update(X1)
    assert wrapped.detector_.fit_count == 1
    assert wrapped.detector_.update_count == 1

    # delta = 10 days >= 5 days -> refits on all 20 seen observations
    wrapped.update(X2)
    assert wrapped.detector_.fit_count == 2
    assert wrapped.detector_.last_fit_len == 20
