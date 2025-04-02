import pytest
import pandas as pd
from sktime.forecasting.mapa import MAPAForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.datasets import load_airline

@pytest.fixture
def data():
    y = load_airline()
    y = y.to_timestamp(how='S').asfreq("MS") 
    fh = ForecastingHorizon([1, 2, 3], is_relative=True, freq="MS")
    return y, fh


def test_mapaforecaster_predict(data):
    y, fh = data
    forecaster = MAPAForecaster()
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict(fh=fh)
    expected_fh = fh.to_absolute(y.index[-1])
    expected_idx = expected_fh.to_pandas() 
    assert y_pred.index.equals(expected_idx), "Prediction index mismatch"

def test_mapaforecaster_fit(data):
    """Test MAPAForecaster's fit method with pandas 1.x compatibility."""
    y, fh = data
    forecaster = MAPAForecaster()
    result = forecaster.fit(y, fh=fh)
    assert result is forecaster, "Fit method does not return self."
