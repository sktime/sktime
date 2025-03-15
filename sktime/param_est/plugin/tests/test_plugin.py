# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for plugin composites for parameter estimators."""

import pandas as pd
import pytest

from sktime.param_est.seasonality import SeasonalityACF
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_plugin_fcst():
    """Test PluginParamsForecaster - same as docstring."""
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.param_est.fixed import FixedParams
    from sktime.param_est.plugin import PluginParamsForecaster
    from sktime.transformations.series.difference import Differencer

    y = load_airline()

    # sp_est is a seasonality estimator
    # ACF assumes stationarity so we concat with differencing first
    sp_est = Differencer() * SeasonalityACF()

    # fcst is a forecaster with a "sp" parameter which we want to tune
    fcst = NaiveForecaster()

    # sp_auto is auto-tuned via PluginParamsForecaster
    sp_auto = PluginParamsForecaster(sp_est, fcst)

    # fit sp_auto to data, predict, and inspect the tuned sp parameter
    sp_auto.fit(y, fh=[1, 2, 3])
    y_pred = sp_auto.predict()
    assert isinstance(y_pred, pd.Series)

    assert sp_auto.forecaster_.get_params()["sp"] == 12

    # shorthand ways to specify sp_auto, via dunder, does the same
    sp_auto2 = sp_est * fcst
    assert isinstance(sp_auto2, PluginParamsForecaster)
    assert sp_auto2 == sp_auto

    # or entire pipeline in one go
    sp_auto3 = Differencer() * SeasonalityACF() * NaiveForecaster()
    assert isinstance(sp_auto3, PluginParamsForecaster)
    assert sp_auto3 == sp_auto

    # plugin with dict
    sp_plugin = PluginParamsForecaster(
        FixedParams({"foo": 42}), NaiveForecaster(), params={"sp": "foo"}
    )

    sp_plugin.fit(y, fh=[1, 2, 3])
    assert sp_plugin.forecaster_.get_params()["sp"] == 42


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_plugin_trafo():
    """Test PluginParamsTransformer - same as docstring."""
    from sktime.datasets import load_airline
    from sktime.param_est.fixed import FixedParams
    from sktime.param_est.plugin import PluginParamsTransformer
    from sktime.param_est.seasonality import SeasonalityACF
    from sktime.transformations.series.detrend import Deseasonalizer
    from sktime.transformations.series.difference import Differencer

    X = load_airline()

    # sp_est is a seasonality estimator
    # ACF assumes stationarity so we concat with differencing first
    sp_est = Differencer() * SeasonalityACF()

    # trafo is a forecaster with a "sp" parameter which we want to tune
    trafo = Deseasonalizer()
    sp_auto = PluginParamsTransformer(sp_est, trafo)

    # fit sp_auto to data, transform, and inspect the tuned sp parameter
    sp_auto.fit(X)

    Xt = sp_auto.transform(X)
    assert isinstance(Xt, pd.Series)
    assert sp_auto.transformer_.get_params()["sp"] == 12

    # shorthand ways to specify sp_auto, via dunder, does the same
    sp_auto2 = sp_est * trafo
    assert isinstance(sp_auto2, PluginParamsTransformer)
    assert sp_auto2 == sp_auto

    # or entire pipeline in one go
    sp_auto3 = Differencer() * SeasonalityACF() * Deseasonalizer()
    assert isinstance(sp_auto3, PluginParamsTransformer)
    assert sp_auto3 == sp_auto

    # plugin with dict
    sp_plugin = PluginParamsTransformer(
        FixedParams({"foo": 42}), Deseasonalizer(), params={"sp": "foo"}
    )

    sp_plugin.fit(X)
    assert sp_plugin.transformer_.get_params()["sp"] == 42
