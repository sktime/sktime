"""Tests for PyKAN defaults and effective lag-window resolution."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.pykan import PyKANForecaster

pytestmark = pytest.mark.skipif(
    not _check_estimator_deps(PyKANForecaster, severity="none"),
    reason="run only when PyKAN dependencies are present",
)


def _fast_forecaster(**kwargs):
    """Use a real KAN model with a small budget and no checkpoint writes."""
    return PyKANForecaster(
        grids=[2],
        model_params={"k": 2, "auto_save": False, "symbolic_enabled": False},
        fit_params={"steps": 1},
        **kwargs,
    )


@pytest.mark.parametrize("n,horizon,expected", [(40, 3, 10), (12, 2, 3), (10, 3, 1)])
def test_default_lags_adapt_to_available_history(n, horizon, expected):
    """Default lag resolution leaves a complete window in both partitions."""
    y = pd.Series(np.sin(np.arange(n)), name="target")
    forecaster = _fast_forecaster().fit(y, fh=list(range(1, horizon + 1)))
    assert forecaster.input_layer_size is None
    assert forecaster._input_layer_size == expected
    assert forecaster._layer_sizes == [expected, 5, horizon]
    prediction = forecaster.predict()
    assert len(prediction) == horizon
    assert np.isfinite(prediction).all()
    assert forecaster.clone().input_layer_size is None


def test_explicit_lags_and_hidden_widths_are_retained():
    """Existing explicit architectures remain available and cloneable."""
    y = pd.Series(np.sin(np.arange(40)))
    forecaster = _fast_forecaster(input_layer_size=2, hidden_layers=(1, 1))
    forecaster.fit(y, fh=[1, 2])
    assert forecaster._layer_sizes == [2, 1, 1, 2]
    assert forecaster.get_params()["fit_params"] == {"steps": 1}


def test_default_training_budget_runs_both_grids():
    """The public default performs training rather than a one-step smoke fit."""
    y = pd.Series(np.sin(2 * np.pi * np.arange(60) / 12))
    forecaster = PyKANForecaster(
        model_params={"k": 2, "auto_save": False, "symbolic_enabled": False}
    ).fit(y, fh=[1, 2, 3])
    assert forecaster.fit_params is None
    assert len(forecaster.train_losses) == len(forecaster.test_losses) == 40
    assert np.isfinite(forecaster.predict()).all()


@pytest.mark.parametrize("lags", [0, -1, 1.5, True, 100])
def test_invalid_or_infeasible_lags_raise_clear_error(lags):
    """Fail before building empty tensors or an invalid model."""
    with pytest.raises(ValueError, match="input_layer_size"):
        _fast_forecaster(input_layer_size=lags).fit(
            pd.Series(np.arange(12.0)), fh=[1, 2]
        )


def test_too_short_default_window_raises_clear_error():
    """A horizon exceeding the partition size cannot be repaired by fewer lags."""
    with pytest.raises(ValueError, match="enough data"):
        _fast_forecaster().fit(pd.Series(np.arange(6.0)), fh=[1, 2, 3])


def test_exogenous_features_do_not_inflate_target_lookback():
    """Model width includes future covariates while target history uses only lags."""
    y = pd.Series(np.sin(np.arange(40.0)))
    X = pd.DataFrame({"x": np.cos(np.arange(42.0))})
    forecaster = _fast_forecaster().fit(y, X=X.iloc[:40], fh=[1, 2])
    assert forecaster._input_layer_size == 10
    assert forecaster._layer_sizes[0] == 12
    prediction = forecaster.predict(X=X.iloc[40:])
    assert len(prediction) == 2
    assert np.isfinite(prediction).all()
