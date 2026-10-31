"""Unit tests for ChronosForecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.chronos import ChronosForecaster


class _DummyConfig:
    context_length = 4


class _DummyModel:
    config = _DummyConfig()


class _DummyPipeline:
    model = _DummyModel()


def test_chronos_update_restricts_context_to_context_length():
    """``_update`` retains only the latest model ``context_length``."""
    y = pd.DataFrame({"y": np.arange(20, dtype=float)})
    forecaster = ChronosForecaster(model_path="dummy/chronos", ignore_deps=True)
    forecaster._load_pipeline = lambda: _DummyPipeline()
    forecaster._fit(y)

    y_new = pd.DataFrame(
        {"y": np.arange(20, 25, dtype=float)}, index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert len(forecaster._context) == 4
    np.testing.assert_array_equal(forecaster._context["y"].to_numpy(), [21, 22, 23, 24])
