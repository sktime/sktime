"""Unit tests for TimesFM2Forecaster context-limited update."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from sktime.forecasting.timesfm2 import TimesFM2Forecaster


def test_timesfm2_update_restricts_context_to_model_context_length():
    """``_update`` trims ``context_`` to the model ``context_length``."""
    y = pd.Series(np.arange(20, dtype=float))
    forecaster = TimesFM2Forecaster(model_path=None, config={"context_length": 4})
    forecaster._load_model = lambda: SimpleNamespace(
        config=SimpleNamespace(context_length=4, horizon_length=8)
    )
    forecaster._fit(y)

    y_new = pd.Series(np.arange(20, 25, dtype=float), index=pd.RangeIndex(20, 25))
    forecaster._update(y_new)

    assert len(forecaster.context_) == 4
    np.testing.assert_array_equal(forecaster.context_.to_numpy(), [21, 22, 23, 24])
