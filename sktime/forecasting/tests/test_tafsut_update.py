"""Unit tests for TafsutForecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.tafsut import TafsutForecaster


def test_tafsut_update_restricts_context_to_context_length():
    """``_update`` retains only the latest configured ``context_length``."""
    y = pd.Series(np.arange(20, dtype=float), name="y")
    forecaster = TafsutForecaster(
        model_path=None, config={"context_length": 4}, device="cpu"
    )
    forecaster.load = lambda: type("M", (), {"cfg": type("C", (), {"context_length": 4})()})()
    forecaster._fit(y)

    y_new = pd.Series(np.arange(20, 25, dtype=float), index=pd.RangeIndex(20, 25), name="y")
    forecaster._update(y_new)

    assert len(forecaster._context) == 4
    np.testing.assert_array_equal(forecaster._context.to_numpy(), [21, 22, 23, 24])
