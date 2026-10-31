"""Unit tests for TimesFM3Forecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.timesfm3 import TimesFM3Forecaster


def test_timesfm3_update_restricts_history_to_global_context():
    """``_update`` retains only the latest ``global_context`` observations."""
    y = pd.DataFrame({"y": np.arange(20, dtype=float)})
    forecaster = TimesFM3Forecaster(license_accepted=True, ignore_deps=True)
    forecaster._check_license = lambda: None
    dummy = type("M", (), {"global_context": 4})()
    forecaster._load_model = lambda: dummy
    forecaster._ensure_model_loaded = lambda: dummy
    # simulate public fit retaining y
    forecaster._y = y
    forecaster._fit(y, X=None, fh=None)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=float)}, index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert len(forecaster._y) == 4
    np.testing.assert_array_equal(forecaster._y["y"].to_numpy(), [21, 22, 23, 24])
