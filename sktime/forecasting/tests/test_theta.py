"""Tests for ThetaForecaster."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["big-o", "kejsitake", "ciaran-g"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.theta import ThetaForecaster, ThetaModularForecaster
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.validation.forecasting import check_fh


@pytest.mark.skipif(
    not run_test_for_class(ThetaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predictive_performance_on_airline():
    """Check prediction performance on airline dataset.

    Performance on this dataset should be reasonably good.

    Raises
    ------
    AssertionError - if point forecasts do not lie close to the test data
    """
    y = np.log1p(load_airline())
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(len(y_test)) + 1

    f = ThetaForecaster(sp=12)
    f.fit(y_train)
    y_pred = f.predict(fh=fh)

    np.testing.assert_allclose(y_pred, y_test, rtol=0.05)


@pytest.mark.skipif(
    not run_test_for_class(ThetaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_pred_errors_against_y_test(fh):
    """Check prediction performance on airline dataset.

    Y_test must lie in the prediction interval with coverage=0.9.

    Arguments
    ---------
    fh: ForecastingHorizon, fh at which to test prediction

    Raises
    ------
    AssertionError - if point forecasts do not lie within the prediction intervals
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    f = ThetaForecaster()
    f.fit(y_train, fh=fh)

    intervals = f.predict_interval(fh=fh, coverage=0.9)

    y_test = y_test.iloc[check_fh(fh) - 1]

    # Performance should be good enough that all point forecasts lie within the
    # prediction intervals.
    assert np.all(y_test > intervals[(y.name, 0.9, "lower")].values)
    assert np.all(y_test < intervals[(y.name, 0.9, "upper")].values)


@pytest.mark.skipif(
    not run_test_for_class(ThetaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecaster_with_initial_level():
    """Check prediction performance on airline dataset.

    Performance on this dataset should be reasonably good.

    Raises
    ------
    AssertionError - if point forecasts do not lie close to the test data
    """
    y = np.log1p(load_airline())
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(len(y_test)) + 1

    f = ThetaForecaster(initial_level=0.1, sp=12)
    f.fit(y_train)
    y_pred = f.predict(fh=fh)

    np.testing.assert_allclose(y_pred, y_test, rtol=0.05)


@pytest.mark.skipif(
    not run_test_for_class([ThetaForecaster, ThetaModularForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_theta_and_thetamodular():
    """Check predictions ThetaForecaster and ThetaModularForecaster align.

    Raises
    ------
    AssertionError - if point forecasts of Theta and ThetaModular do not lie
    close to each other.
    """
    y = np.log1p(load_airline())
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(len(y_test)) + 1

    f = ThetaForecaster(sp=12)
    f.fit(y_train)
    y_pred_theta = f.predict(fh=fh)

    f1 = ThetaModularForecaster(theta_values=(0, 2))
    f1.fit(y_train)
    y_pred_thetamodular = f1.predict(fh=fh)

    np.testing.assert_allclose(y_pred_theta, y_pred_thetamodular, rtol=0.06)


@pytest.mark.skipif(
    not run_test_for_class(ThetaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def check_panel_theta_quantiles():
    """Test predict quantiles with theta on panel data with datetime index."""
    # make panel with hour of day panel and datetime index
    y = load_airline()
    y.index = pd.date_range(start="1960-01-01", periods=len(y.index), freq="H")
    y.index.names = ["datetime"]
    y.name = "passengers"
    y = y.to_frame()
    y["hour_of_day"] = y.index.hour
    y = y.reset_index().set_index(["hour_of_day", "datetime"]).sort_index()

    forecaster = ThetaForecaster(sp=1)
    forecaster.fit(y)
    forecaster.predict(fh=[1, 3])
    forecaster.predict_quantiles(fh=[1, 3], alpha=[0.1, 0.5, 0.9])
