"""Unit tests for WindFMForecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.windfm import WindFMForecaster


def test_windfm_update_restricts_context_to_max_context():
    """``_update`` retains only the latest ``max_context_`` observations."""
    y = pd.Series(np.arange(20, dtype=float), name="power")
    X = pd.DataFrame(
        {
            "wind_speed": np.arange(20, dtype=float),
            "wind_direction": np.arange(20, dtype=float),
            "density": np.arange(20, dtype=float),
            "temperature": np.arange(20, dtype=float),
            "pressure": np.arange(20, dtype=float),
        }
    )
    forecaster = WindFMForecaster()
    forecaster._load_windfm = lambda: (object(), object())
    forecaster._fit(y, X=X)
    # emulate a shorter retained window than fit length
    forecaster.max_context_ = 4

    y_new = pd.Series(np.arange(20, 25, dtype=float), index=pd.RangeIndex(20, 25), name="power")
    X_new = pd.DataFrame(
        {
            "wind_speed": np.arange(20, 25, dtype=float),
            "wind_direction": np.arange(20, 25, dtype=float),
            "density": np.arange(20, 25, dtype=float),
            "temperature": np.arange(20, 25, dtype=float),
            "pressure": np.arange(20, 25, dtype=float),
        },
        index=pd.RangeIndex(20, 25),
    )
    forecaster._update(y_new, X=X_new)

    assert len(forecaster.y_context_) == 4
    np.testing.assert_array_equal(forecaster.y_context_.to_numpy(), [21, 22, 23, 24])
    assert len(forecaster.X_context_) == 4
