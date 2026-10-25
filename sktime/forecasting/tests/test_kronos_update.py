"""Unit tests for KronosForecaster context-limited update."""

import numpy as np
import pandas as pd

from sktime.forecasting.kronos import KronosForecaster


def test_kronos_update_restricts_context_to_max_context():
    """``_update`` trims ``context_`` to fit-time ``max_context_``."""
    y = pd.DataFrame({"close": np.arange(10, dtype=float)})
    forecaster = KronosForecaster()
    forecaster._load_kronos = lambda: (object(), object())
    forecaster._resolve_column_mapping = lambda y: {"close": "close"}
    # Drive _fit body without network: set fields as _fit would
    forecaster.context_ = y.copy()
    forecaster.max_context_ = 4
    forecaster.column_mapping_ = {"close": "close"}
    forecaster.output_columns_ = ["close"]
    forecaster.tokenizer_, forecaster.model_ = object(), object()

    y_new = pd.DataFrame(
        {"close": np.arange(10, 14, dtype=float)}, index=pd.RangeIndex(10, 14)
    )
    forecaster._update(y_new)

    assert len(forecaster.context_) == 4
    np.testing.assert_array_equal(
        forecaster.context_["close"].to_numpy(), [10, 11, 12, 13]
    )
