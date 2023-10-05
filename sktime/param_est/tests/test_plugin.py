# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for parameter plugin transformers."""

__author__ = ["fkiraly"]

import numpy as np
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.naive import NaiveForecaster
from sktime.param_est.fixed import FixedParams
from sktime.param_est.plugin import PluginParamsForecaster
from sktime.param_est.seasonality import SeasonalityACF
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.difference import Differencer
from sktime.utils.validation._dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_seasonality_acf():
    """Test PluginParamsForecaster on airline data.

    Same as docstring example.
    """
    y = load_airline()

    sp_est = Differencer() * SeasonalityACF()
    fcst = NaiveForecaster()
    sp_auto = PluginParamsForecaster(sp_est, fcst)
    sp_auto.fit(y, fh=[1, 2, 3])
    assert sp_auto.forecaster_.get_params()["sp"] == 12

    PluginParamsForecaster(
        FixedParams({"foo": 12}), NaiveForecaster(), params={"foo": "sp"}
    )


@pytest.mark.skipif(
    not run_test_for_class(PluginParamsForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_paramplugin_dict():
    """Test PluginParamsForecaster with param: dict.

    Failure case from bug report #4921.
    """
    from sklearn.ensemble import RandomForestRegressor

    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.compose import EnsembleForecaster, make_reduction
    from sktime.forecasting.model_selection import ForecastingGridSearchCV
    from sktime.split import ExpandingWindowSplitter

    y, X = load_longley()
    horizon = ForecastingHorizon(np.arange(1, 4), is_relative=True)

    random_forest = make_reduction(
        RandomForestRegressor(),
        window_length=2,
        strategy="direct",
        windows_identical=False,
    )
    expanding_window_cv = ExpandingWindowSplitter(fh=horizon, step_length=1)
    cv_random_forest = ForecastingGridSearchCV(
        forecaster=random_forest,
        cv=expanding_window_cv,
        param_grid={"estimator__max_features": np.linspace(0.1, 0.9, num=5)},
        return_n_best_forecasters=3,
    )

    ensembler = EnsembleForecaster(forecasters=[], aggfunc="median")
    plugin_fcst = PluginParamsForecaster(
        param_est=cv_random_forest,
        forecaster=ensembler,
        params={"forecasters": "n_best_forecasters"},
    )

    plugin_fcst.fit(y, X, fh=horizon)
    plugin_fcst.predict(fh=horizon, X=X)
