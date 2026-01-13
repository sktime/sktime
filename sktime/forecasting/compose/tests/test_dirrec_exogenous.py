"""Tests for DirRec with exogenous variables."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import DirRecTabularRegressionForecaster


def test_dirrec_with_exogenous_basic():
    """Test that DirRecTabularRegressionForecaster works with exogenous variables."""
    # Create simple data
    n = 30
    y = pd.Series(np.arange(n), index=pd.date_range("2020-01-01", periods=n, freq="D"))
    X = pd.DataFrame({"x1": np.arange(n) * 2, "x2": np.arange(n) * 3}, index=y.index)

    # Split train/test
    y_train = y.iloc[:-3]
    X_train, X_test = X.iloc[:-3], X.iloc[-3:]

    # Fit with exogenous
    fh = ForecastingHorizon([1, 2, 3])
    forecaster = DirRecTabularRegressionForecaster(
        estimator=LinearRegression(), window_length=5
    )
    forecaster.fit(y_train, X=X_train, fh=fh)
    y_pred = forecaster.predict(fh=fh, X=X_test)

    # Check shape
    assert len(y_pred) == 3
    assert not y_pred.isna().any()


def test_dirrec_exogenous_vs_no_exogenous():
    """Test that exogenous variables affect predictions."""
    n = 40
    # Create data where X helps predict y
    x_vals = np.random.RandomState(42).randn(n)
    y_vals = x_vals * 2 + np.random.RandomState(42).randn(n) * 0.1

    y = pd.Series(y_vals, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    X = pd.DataFrame({"x1": x_vals}, index=y.index)

    y_train = y.iloc[:-3]
    X_train, X_test = X.iloc[:-3], X.iloc[-3:]

    fh = ForecastingHorizon([1, 2, 3])

    # Fit with X
    fc_with_x = DirRecTabularRegressionForecaster(
        estimator=LinearRegression(), window_length=5
    )
    fc_with_x.fit(y_train, X=X_train, fh=fh)
    pred_with_x = fc_with_x.predict(fh=fh, X=X_test)

    # Fit without X
    fc_no_x = DirRecTabularRegressionForecaster(
        estimator=LinearRegression(), window_length=5
    )
    fc_no_x.fit(y_train, fh=fh)
    pred_no_x = fc_no_x.predict(fh=fh)

    # Predictions should be different
    assert not np.allclose(pred_with_x.values, pred_no_x.values, atol=0.01)


def test_dirrec_multivariate_exogenous():
    """Test DirRec with multiple exogenous variables."""
    n = 35
    y = pd.Series(np.arange(n), index=pd.date_range("2020-01-01", periods=n, freq="D"))
    X = pd.DataFrame(
        {
            "x1": np.arange(n),
            "x2": np.arange(n) ** 2,
            "x3": np.sin(np.arange(n)),
        },
        index=y.index,
    )

    y_train = y.iloc[:-3]
    X_train, X_test = X.iloc[:-3], X.iloc[-3:]

    fh = ForecastingHorizon([1, 2, 3])
    forecaster = DirRecTabularRegressionForecaster(
        estimator=LinearRegression(), window_length=7
    )
    forecaster.fit(y_train, X=X_train, fh=fh)
    y_pred = forecaster.predict(fh=fh, X=X_test)

    assert len(y_pred) == 3
    assert not y_pred.isna().any()
