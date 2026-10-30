"""Tests for pretrained state handling in forecasting tuners."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["kr1xh"]

import pytest

from sktime.forecasting.dummy_global import DummyGlobalForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import SingleWindowSplitter
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


def _make_pretrained_dummy():
    """Return a DummyGlobalForecaster pretrained on a small panel."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(3,),
        min_timepoints=10,
        max_timepoints=10,
        n_columns=1,
        random_state=0,
    )
    forecaster = DummyGlobalForecaster()
    forecaster.pretrain(y_panel)
    return forecaster


def test_grid_search_keeps_pretrained_state():
    """Tuner candidates and best forecaster retain pretrained attributes."""
    forecaster = _make_pretrained_dummy()
    pretrain_mean = forecaster.global_mean_

    y = _make_series(n_timepoints=30, random_state=1)
    # precondition: the test can tell pretrain mean from the fit-series mean
    assert pretrain_mean != pytest.approx(y.mean())

    cv = SingleWindowSplitter(fh=[1, 2, 3], window_length=20)
    gscv = ForecastingGridSearchCV(
        forecaster,
        cv=cv,
        param_grid={"strategy": ["mean", "last"]},
        scoring=MeanSquaredError(),
    )
    gscv.fit(y)

    # if set_params wiped pretrained state, _fit would recompute the mean from y
    assert gscv.best_forecaster_.global_mean_ == pytest.approx(pretrain_mean)
    for _, candidate in gscv.n_best_forecasters_:
        assert candidate.global_mean_ == pytest.approx(pretrain_mean)

    # the user's forecaster must not be mutated by the search
    assert forecaster.state == "pretrained"
    assert forecaster.strategy == "mean"
    assert forecaster.global_mean_ == pytest.approx(pretrain_mean)
