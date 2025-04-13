"""Tests for AutoETS predict_quantiles with different forecasting horizons."""

__author__ = ["nahcol10"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from sktime.tests.test_switch import run_test_for_class

airline = load_airline()

y_train = airline[:60]

ALPHA_LEVELS = [0.05, 0.5, 0.95]

# Define different types of forecasting horizons for testing
FH_CASES = [
    1,
    np.arange(1, 4),
    [1, 3, 5],
    [-2, 0, 2],
    np.arange(-3, 4),
    -1,
]


@pytest.fixture
def auto_true_model():
    """Fixture for AutoETS model with auto=True."""
    forecaster = AutoETS(auto=True, sp=12, n_jobs=-1)
    forecaster.fit(y_train)
    return forecaster


@pytest.fixture
def auto_false_model():
    """Fixture for AutoETS model with auto=False."""
    forecaster = AutoETS(
        auto=False, sp=12, n_jobs=-1, error="add", trend="add", seasonal="add"
    )
    forecaster.fit(y_train)
    return forecaster


@pytest.mark.skipif(
    not run_test_for_class(AutoETS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("fh_input", FH_CASES)
def test_predict_quantiles_auto_true(auto_true_model, fh_input):
    """Test predict_quantiles with auto=True for different forecasting horizons."""
    fh = ForecastingHorizon(fh_input)
    pred_quantiles = auto_true_model.predict_quantiles(fh=fh, alpha=ALPHA_LEVELS)

    assert isinstance(pred_quantiles, pd.DataFrame)
    assert len(pred_quantiles) == len(fh)
    assert pred_quantiles.columns.nlevels == 2
    assert len(pred_quantiles.columns.get_level_values(1).unique()) == len(ALPHA_LEVELS)

    var_name = pred_quantiles.columns.get_level_values(0)[0]
    for i in range(len(pred_quantiles)):
        values = [pred_quantiles.iloc[i][(var_name, alpha)] for alpha in ALPHA_LEVELS]
        assert all(values[j] <= values[j + 1] for j in range(len(values) - 1))


@pytest.mark.skipif(
    not run_test_for_class(AutoETS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("fh_input", FH_CASES)
def test_predict_quantiles_auto_false(auto_false_model, fh_input):
    """Test predict_quantiles with auto=False for different forecasting horizons."""
    fh = ForecastingHorizon(fh_input)
    pred_quantiles = auto_false_model.predict_quantiles(fh=fh, alpha=ALPHA_LEVELS)

    assert isinstance(pred_quantiles, pd.DataFrame)
    assert len(pred_quantiles) == len(fh)

    assert pred_quantiles.columns.nlevels == 2
    assert len(pred_quantiles.columns.get_level_values(1).unique()) == len(ALPHA_LEVELS)

    var_name = pred_quantiles.columns.get_level_values(0)[0]
    for i in range(len(pred_quantiles)):
        values = [pred_quantiles.iloc[i][(var_name, alpha)] for alpha in ALPHA_LEVELS]
        assert all(values[j] <= values[j + 1] for j in range(len(values) - 1))
