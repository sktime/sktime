"""Unit tests for TiRex2Forecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.tirex2 import TiRex2Forecaster


def test_tirex2_update_restricts_context_to_context_len():
    """``_update`` retains only the latest model ``context_len``."""
    y = pd.DataFrame({"y": np.arange(20, dtype=float)})
    forecaster = TiRex2Forecaster(ignore_deps=True)
    dummy = type("M", (), {"context_len": 4})()
    forecaster._ensure_model_loaded = lambda: dummy
    forecaster._fit(y)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=float)}, index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert len(forecaster._context) == 4
    np.testing.assert_array_equal(forecaster._context["y"].to_numpy(), [21, 22, 23, 24])
