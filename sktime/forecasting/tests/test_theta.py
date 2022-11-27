# -*- coding: utf-8 -*-
"""Tests for ThetaForecaster.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["@big-o", "kejsitake"]

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.theta import ThetaForecaster, ThetaModularForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_fh


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
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
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
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
    AssertionError - if point forecasts do not lie withing the prediction intervals
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    f = ThetaForecaster()
    f.fit(y_train, fh=fh)

    intervals = f.predict_interval(fh=fh, coverage=0.9)

    y_test = y_test.iloc[check_fh(fh) - 1]

    # Performance should be good enough that all point forecasts lie within the
    # prediction intervals.
    assert np.all(y_test > intervals[("Coverage", 0.9, "lower")].values)
    assert np.all(y_test < intervals[("Coverage", 0.9, "upper")].values)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
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
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
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
