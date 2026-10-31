"""Unit tests for AuroraForecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.aurora import AuroraForecaster


def test_aurora_update_restricts_context_to_context_length():
    """``_update`` retains only the latest configured ``context_length``."""
    y = pd.DataFrame({"y": np.arange(20, dtype=float)})
    forecaster = AuroraForecaster(context_length=4)
    forecaster._load_model = lambda: type("M", (), {"eval": lambda self: None})()
    forecaster._fit(y)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=float)}, index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert len(forecaster._context) == 4
    np.testing.assert_array_equal(forecaster._context["y"].to_numpy(), [21, 22, 23, 24])
