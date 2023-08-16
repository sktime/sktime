"""Tests the SARIMAX model."""
__author__ = ["TNTran92", "yarnabrina"]

import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from sktime.forecasting.sarimax import SARIMAX
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.skipif(
    not run_test_for_class(SARIMAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SARIMAX_against_statsmodels():
    """Compares Sktime's and Statsmodel's SARIMAX."""
    from statsmodels.tsa.api import SARIMAX as _SARIMAX

    df = make_forecasting_problem()

    sktime_model = SARIMAX(order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    sktime_model.fit(df)
    y_pred = sktime_model.predict(df.index)

    stats = _SARIMAX(endog=df, order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    stats_fit = stats.fit()
    stats_pred = stats_fit.predict(df.index[0])
    assert_allclose(y_pred.tolist(), stats_pred.tolist())


@pytest.mark.skipif(
    not run_test_for_class(SARIMAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SARIMAX_single_interval_against_statsmodels():
    """Compares Sktime's and Statsmodel's SARIMAX.

    Notes
    -----
    * Predict confidence intervals using underlying estimator and the wrapper.
    * Predicts for a single coverage.
    * Uses a non-default value of 97.5% to test inputs are actually being respected.
    """
    from statsmodels.tsa.api import SARIMAX as _SARIMAX

    df = make_forecasting_problem()

    sktime_model = SARIMAX(order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    sktime_model.fit(df)
    sktime_pred_int = sktime_model.predict_interval(df.index, coverage=0.975)
    sktime_pred_int = sktime_pred_int.xs((0, 0.975), axis="columns")

    stats = _SARIMAX(endog=df, order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    stats_fit = stats.fit()
    stats_pred_int = stats_fit.get_prediction(df.index[0]).conf_int(alpha=0.025)
    stats_pred_int.columns = ["lower", "upper"]

    assert_frame_equal(sktime_pred_int, stats_pred_int)


@pytest.mark.skipif(
    not run_test_for_class(SARIMAX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SARIMAX_multiple_intervals_against_statsmodels():
    """Compares Sktime's and Statsmodel's SARIMAX.

    Notes
    -----
    * Predict confidence intervals using underlying estimator and the wrapper.
    * Predicts for multiple coverage values, viz. 70% and 80%.
    """
    from statsmodels.tsa.api import SARIMAX as _SARIMAX

    df = make_forecasting_problem()

    sktime_model = SARIMAX(order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    sktime_model.fit(df)
    sktime_pred_int = sktime_model.predict_interval(df.index, coverage=[0.70, 0.80])
    sktime_pred_int_70 = sktime_pred_int.xs((0, 0.70), axis="columns")
    sktime_pred_int_80 = sktime_pred_int.xs((0, 0.80), axis="columns")

    stats = _SARIMAX(endog=df, order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))
    stats_fit = stats.fit()
    stats_pred_int_70 = stats_fit.get_prediction(df.index[0]).conf_int(alpha=0.30)
    stats_pred_int_70.columns = ["lower", "upper"]
    stats_pred_int_80 = stats_fit.get_prediction(df.index[0]).conf_int(alpha=0.20)
    stats_pred_int_80.columns = ["lower", "upper"]

    assert_frame_equal(sktime_pred_int_70, stats_pred_int_70)
    assert_frame_equal(sktime_pred_int_80, stats_pred_int_80)
