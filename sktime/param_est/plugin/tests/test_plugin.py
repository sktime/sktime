# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for plugin composites for parameter estimators."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.param_est.seasonality import SeasonalityACF


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
    from sktime.transformations.difference import Differencer

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
    from sktime.transformations.detrend import Deseasonalizer
    from sktime.transformations.difference import Differencer

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


@pytest.mark.parametrize(
    "param_est_params, params",
    [
        ({"foo": 12}, {"sp": "foo"}),
        ({"sp": 12}, {"sp": "sp"}),
        ({"sp": 12}, "sp"),
        ({"sp": 12}, ["sp"]),
        ({"sp": 12}, None),
    ],
)
def test_plugin_fcst_update_params(param_est_params, params):
    """Test that update with update_params=True equals fit on all data (#12004)."""
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.param_est.fixed import FixedParams
    from sktime.param_est.plugin import PluginParamsForecaster

    y = load_airline()
    y_train, y_new1, y_new2 = y.iloc[:100], y.iloc[100:112], y.iloc[112:124]

    def make():
        return PluginParamsForecaster(
            FixedParams(param_est_params),
            NaiveForecaster(),
            params=params,
            update_params=True,
        )

    updated = make().fit(y_train, fh=[1, 2, 3])
    updated.update(y_new1)
    updated.update(y_new2)

    refit = make().fit(y.iloc[:124], fh=[1, 2, 3])

    assert updated.forecaster_.get_params()["sp"] == 12
    assert updated.cutoff == refit.cutoff
    np.testing.assert_allclose(updated.predict(), refit.predict())


def test_plugin_fcst_update_params_exogenous():
    """Test that update with update_params=True pools X as well (#12004)."""
    from sklearn.linear_model import LinearRegression

    from sktime.datasets import load_longley
    from sktime.forecasting.compose import YfromX
    from sktime.param_est.fixed import FixedParams
    from sktime.param_est.plugin import PluginParamsForecaster

    y, X = load_longley()

    def make():
        return PluginParamsForecaster(
            FixedParams({}), YfromX(LinearRegression()), update_params=True
        )

    updated = make().fit(y.iloc[:10], X=X.iloc[:10], fh=[1])
    updated.update(y.iloc[10:13], X=X.iloc[10:13])

    refit = make().fit(y.iloc[:13], X=X.iloc[:13], fh=[1])

    X_future = X.iloc[13:14]
    np.testing.assert_allclose(updated.predict(X=X_future), refit.predict(X=X_future))
