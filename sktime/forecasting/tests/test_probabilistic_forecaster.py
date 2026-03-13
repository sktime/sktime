import pandas as pd
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.probabilistic_forecaster import ProbabilisticIntermittentForecaster

def test_poisson_forecaster():
    y = pd.Series([0, 1, 0, 2, 0, 3])
    fh = ForecastingHorizon([1, 2, 3])
    forecaster = ProbabilisticIntermittentForecaster(model_type="poisson")
    forecaster.fit(y)
    y_pred = forecaster.predict(fh)
    assert len(y_pred) == 3
    assert all(y_pred >= 0)

def test_nb_hurdle_forecaster():
    y = pd.Series([0, 1, 0, 2, 0, 3])
    fh = ForecastingHorizon([1, 2, 3])
    forecaster = ProbabilisticIntermittentForecaster(model_type="nb_hurdle")
    forecaster.fit(y)
    y_pred = forecaster.predict(fh)
    assert len(y_pred) == 3
    assert all(y_pred >= 0)
