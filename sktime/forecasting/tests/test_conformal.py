"""Tests for the ConformalIntervals probability wrapper."""
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import check_is_mtype
from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class

__author__ = ["fkiraly"]


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_standard():
    """Tests standard use of the conformal intervals adapter."""
    y = load_airline()
    forecaster = NaiveForecaster(strategy="drift")

    conformal_forecaster = ConformalIntervals(forecaster)
    conformal_forecaster.fit(y, fh=[1, 2, 3])
    pred_int = conformal_forecaster.predict_interval()

    assert check_is_mtype(pred_int, "pred_interval", "Proba")


@pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_conformal_with_gscv():
    """With ForecastingGridSearchCV and parameter plugin"""
    from sktime.forecasting.model_selection import (
        ExpandingWindowSplitter,
        ForecastingGridSearchCV,
    )
    from sktime.param_est.plugin import PluginParamsForecaster

    y = load_airline()

    # part 1 = grid search
    cv = ExpandingWindowSplitter(fh=[1, 2, 3])
    forecaster = NaiveForecaster()
    param_grid = {"strategy": ["last", "mean", "drift"]}
    gscv = ForecastingGridSearchCV(
        forecaster=forecaster,
        param_grid=param_grid,
        cv=cv,
    )

    # part 2 = plug in results of grid search into conformal intervals estimator
    conformal_with_fallback = ConformalIntervals(NaiveForecaster())
    gscv_with_conformal = PluginParamsForecaster(
        gscv,
        conformal_with_fallback,
        params={"forecaster": "best_forecaster"},
    )

    gscv_with_conformal.fit(y, fh=[1, 2, 3])

    y_pred_quantiles = gscv_with_conformal.predict_quantiles()

    assert check_is_mtype(y_pred_quantiles, "pred_quantiles", "Proba")
