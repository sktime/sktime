"""Tests the SkforecastAutoreg model."""

__author__ = ["Abhay-Lejith"]

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from sktime.forecasting.compose import SkforecastAutoreg
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.skipif(
    not run_test_for_class(SkforecastAutoreg),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SkforecastAutoreg_predict_against_ForecasterAutoreg():
    """Compares the predictions of the sktime adapter against skforecast's"""
    "ForecasterAutoreg."
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from sklearn.linear_model import LinearRegression

    df = make_forecasting_problem(n_timepoints=10)
    fh = [1, 2, 3]

    sktime_model = SkforecastAutoreg(LinearRegression(), 2)
    sktime_model.fit(df)
    sktime_pred = sktime_model.predict(fh)

    skforecast_model = ForecasterAutoreg(LinearRegression(), 2)
    skforecast_model.fit(df)
    skforecast_pred = skforecast_model.predict(3)

    assert_series_equal(sktime_pred, skforecast_pred, check_names=False)


@pytest.mark.skipif(
    not run_test_for_class(SkforecastAutoreg),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SkforecastAutoreg_predict_with_exog_against_ForecasterAutoreg():
    """Compares the predictions using exog of the sktime adapter against skforecast's"""
    "ForecasterAutoreg."
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from sklearn.linear_model import LinearRegression

    y, X = make_forecasting_problem(n_timepoints=10, make_X=True, index_type="range")
    fh = [1, 2, 3]
    X_train = X.head(7)
    X_test = X.tail(3)
    y_train = y.head(7)

    sktime_model = SkforecastAutoreg(LinearRegression(), 2)
    sktime_model.fit(y_train, X_train)
    sktime_pred = sktime_model.predict(fh, X_test)

    # skforecast requires all or none of the column names to be string
    X_train.columns = X_train.columns.astype("str")
    X_test.columns = X_test.columns.astype("str")

    skforecast_model = ForecasterAutoreg(LinearRegression(), 2)
    skforecast_model.fit(y_train, X_train)
    skforecast_pred = skforecast_model.predict(3, exog=X_test)

    assert_series_equal(sktime_pred, skforecast_pred, check_names=False)


@pytest.mark.skipif(
    not run_test_for_class(SkforecastAutoreg),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SkforecastAutoreg_predict_interval_against_ForecasterAutoreg():
    """Compares the predict interval of the sktime adapter against skforecast's
    ForecasterAutoreg.

    Notes
    -----
    * Predict confidence intervals using underlying estimator and the wrapper.
    * Predicts for a single coverage.
    * Uses a non-default value of 80% to test inputs are actually being respected.
    """
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from sklearn.linear_model import LinearRegression

    df = make_forecasting_problem(n_timepoints=10)
    fh = [1, 2, 3]

    sktime_model = SkforecastAutoreg(LinearRegression(), 2)
    sktime_model.fit(df)
    sktime_pred_int = sktime_model.predict_interval(fh, coverage=0.8)

    skforecast_model = ForecasterAutoreg(LinearRegression(), 2)
    skforecast_model.fit(df)
    skforecast_pred_int = skforecast_model.predict_interval(3, interval=[10, 90])
    skforecast_pred_int = skforecast_pred_int.drop(columns="pred")
    multiindex = [[0, 0], [0.8, 0.8], ["lower", "upper"]]
    skforecast_pred_int.columns = pd.MultiIndex.from_arrays(multiindex)

    assert_frame_equal(sktime_pred_int, skforecast_pred_int, check_names=False)


@pytest.mark.skipif(
    not run_test_for_class(SkforecastAutoreg),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_SkforecastAutoreg_predict_quantile_against_ForecasterAutoreg():
    """Compares the predict quantile of the sktime adapter against skforecast's
    ForecasterAutoreg.

    Notes
    -----
    * Predict quantiles using underlying estimator and the wrapper.
    * Predicts for multiple coverage values, viz. 70% and 80%.
    """
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from sklearn.linear_model import LinearRegression

    df = make_forecasting_problem(n_timepoints=10)
    fh = [1, 2, 3]

    sktime_model = SkforecastAutoreg(LinearRegression(), 2)
    sktime_model.fit(df)
    sktime_pred_qtl = sktime_model.predict_quantiles(fh, alpha=[0.7, 0.8])

    skforecast_model = ForecasterAutoreg(LinearRegression(), 2)
    skforecast_model.fit(df)
    skforecast_pred_qtl = skforecast_model.predict_quantiles(3, quantiles=[0.7, 0.8])
    multiindex = [[0, 0], [0.7, 0.8]]
    skforecast_pred_qtl.columns = pd.MultiIndex.from_arrays(multiindex)

    assert_frame_equal(sktime_pred_qtl, skforecast_pred_qtl)
