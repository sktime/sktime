import numpy as np
import pandas as pd

from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon


def test_naive_forecaster_minimal_memory_same_predictions():
    """Minimal memory mode should give same predictions as full mode.

    This is a first, safe invariant: behaviour must not change, only memory usage.
    """
    # simple univariate monthly series
    y = load_airline()
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    # standard mode
    f_full = NaiveForecaster(strategy="last")
    f_full.fit(y)
    y_pred_full = f_full.predict(fh)

    # minimal memory mode (to be implemented)
    f_min = NaiveForecaster(strategy="last", memory_mode="minimal")
    f_min.fit(y)
    y_pred_min = f_min.predict(fh)

    # predictions must be identical
    assert np.allclose(y_pred_full.to_numpy(), y_pred_min.to_numpy())
    assert y_pred_full.index.equals(y_pred_min.index)
