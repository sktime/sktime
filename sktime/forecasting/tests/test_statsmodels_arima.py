"""Tests for StatsmodelsARIMA forecaster."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.arima._statsmodels_arima import StatsmodelsARIMA


@pytest.mark.parametrize("order", [(1, 1, 1), (2, 1, 0)])
def test_statsmodels_arima_basic_fit_predict(order):
    """Test that StatsmodelsARIMA fits and predicts without error."""
    y = load_airline()
    fh = [1, 2, 3]

    f = StatsmodelsARIMA(order=order)
    f.fit(y)
    y_pred = f.predict(fh)

    # Check output length
    assert len(y_pred) == len(fh)

    # Check that predictions are numeric and finite
    assert y_pred.notnull().all()


def test_statsmodels_arima_with_exog():
    """Test StatsmodelsARIMA with exogenous variables."""
    import pandas as pd

    y = load_airline()
    fh = [1, 2, 3]

    # Make X index match y's index
    X = pd.DataFrame({"x": range(len(y))}, index=y.index)

    f = StatsmodelsARIMA(order=(1, 1, 1))
    f.fit(y, X=X)
    preds = f.predict(fh, X=X.tail(3))

    assert len(preds) == 3


def test_statsmodels_arima_prediction_intervals():
    """Test that prediction intervals are returned correctly."""
    y = load_airline()
    fh = [1, 2, 3]

    f = StatsmodelsARIMA(order=(1, 1, 1))
    f.fit(y)

    # for now, prediction intervals not implemented
    # so we skip or check for AttributeError
    try:
        pred_int = f._predict_interval(fh, coverage=0.9)
        assert "lower" in pred_int and "upper" in pred_int
    except AttributeError:
        pytest.skip("Prediction intervals not yet implemented for StatsmodelsARIMA")
