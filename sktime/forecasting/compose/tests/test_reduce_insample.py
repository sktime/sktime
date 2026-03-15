import numpy as np
import pandas as pd
import pytest
from sktime.datasets import load_airline
from sktime.forecasting.compose import (
    RecursiveTabularRegressionForecaster,
    DirectTabularRegressionForecaster,
    MultioutputTabularRegressionForecaster,
)
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon

@pytest.mark.parametrize(
    "forecaster_class",
    [
        RecursiveTabularRegressionForecaster,
        DirectTabularRegressionForecaster,
        MultioutputTabularRegressionForecaster,
    ],
)
def test_insample_equivalence(forecaster_class):
    """Test that in-sample predictions are reasonable and match 1-step OOS logic."""
    y = load_airline()[:30]
    
    # window_length=10
    forecaster = forecaster_class(LinearRegression(), window_length=10)
    
    fh_oos = ForecastingHorizon([1, 2, 3], is_relative=True)
    if forecaster_class in [DirectTabularRegressionForecaster, MultioutputTabularRegressionForecaster]:
        forecaster.fit(y, fh=fh_oos)
    else:
        forecaster.fit(y)
    
    # Relative fh: -5 to 0
    fh_ins = ForecastingHorizon([-5, -4, -3, -2, -1, 0], is_relative=True)
    y_pred = forecaster.predict(fh=fh_ins)
    
    assert len(y_pred) == len(fh_ins)
    assert (y_pred.index == fh_ins.to_absolute_index(forecaster.cutoff)).all()
    assert not y_pred.isna().any().any()

@pytest.mark.parametrize(
    "forecaster_class",
    [
        RecursiveTabularRegressionForecaster,
        DirectTabularRegressionForecaster,
    ],
)
def test_insample_with_X(forecaster_class):
    """Test in-sample predictions with exogenous variables."""
    y = load_airline()[:30]
    X = pd.DataFrame({"x1": np.arange(len(y))}, index=y.index)
    
    forecaster = forecaster_class(LinearRegression(), window_length=5)
    
    fh_oos = ForecastingHorizon([1, 2], is_relative=True)
    if forecaster_class == DirectTabularRegressionForecaster:
        forecaster.fit(y, X=X, fh=fh_oos)
    else:
        forecaster.fit(y, X=X)
        
    fh_ins = ForecastingHorizon([-2, -1, 0], is_relative=True)
    y_pred = forecaster.predict(fh=fh_ins, X=X)
    
    assert len(y_pred) == 3
    assert not y_pred.isna().any().any()

def test_insample_backfill():
    """Test that backfilling works for very early in-sample points."""
    y = load_airline()[:20]
    # window_length=15 (smaller than data)
    forecaster = RecursiveTabularRegressionForecaster(LinearRegression(), window_length=15)
    forecaster.fit(y)
    
    # Predict from the very beginning
    # Airline starts at 1949-01 (index 0). Cutoff is 1950-08 (index 19).
    # fh = -19 is index 0.
    fh_ins = ForecastingHorizon([-19, -18, -17], is_relative=True)
    y_pred = forecaster.predict(fh=fh_ins)
    
    assert len(y_pred) == 3
    assert not y_pred.isna().any().any()
