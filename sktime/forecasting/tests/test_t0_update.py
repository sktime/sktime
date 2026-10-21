"""Unit tests for T0Forecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.t0 import T0Forecaster


def test_t0_update_restricts_context_to_context_length():
    """``_update`` retains only the latest configured ``context_length``."""
    y = pd.DataFrame({"y": np.arange(20, dtype=float)})
    forecaster = T0Forecaster(context_length=4, license_accepted=True)
    forecaster._check_license = lambda: None
    forecaster._load_model = lambda: object()
    forecaster._fit(y, X=None, fh=None)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=float)}, index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert forecaster.context_.shape == (1, 4)
    np.testing.assert_array_equal(forecaster.context_[0], [21, 22, 23, 24])
