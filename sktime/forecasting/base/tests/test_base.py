# -*- coding: utf-8 -*-
"""Testing advanced functionality of the base class."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from functools import reduce
from operator import mul

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from sktime.datatypes import check_is_mtype, convert
from sktime.datatypes._utilities import get_cutoff, get_window
from sktime.forecasting.arima import ARIMA
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel
from sktime.utils._testing.series import _make_series
from sktime.utils.validation._dependencies import _check_soft_dependencies

PANEL_MTYPES = ["pd-multiindex", "nested_univ", "numpy3D"]
HIER_MTYPES = ["pd_multiindex_hier"]


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("mtype", PANEL_MTYPES)
def test_vectorization_series_to_panel(mtype):
    """Test that forecaster vectorization works for Panel data.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    n_instances = 10

    y = _make_panel(n_instances=n_instances, random_state=42, return_mtype=mtype)

    f = ARIMA()
    y_pred = f.fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(y_pred, mtype, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg

    cutoff_expected = get_cutoff(y)
    msg = (
        "estimator in vectorization test does not properly update cutoff, "
        f"expected {cutoff_expected}, but found {f.cutoff}"
    )
    assert f.cutoff == cutoff_expected, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("mtype", HIER_MTYPES)
def test_vectorization_series_to_hier(mtype):
    """Test that forecaster vectorization works for Hierarchical data.

    This test passes Hierarchical data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    hierarchy_levels = (2, 4)
    n_instances = reduce(mul, hierarchy_levels)

    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)
    y = convert(y, from_type="pd_multiindex_hier", to_type=mtype)

    f = ARIMA()
    y_pred = f.fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(y_pred, mtype, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg

    msg = (
        "estimator in vectorization test does not properly update cutoff, "
        f"expected {y}, but found {f.cutoff}"
    )
    assert f.cutoff == get_cutoff(y), msg


PROBA_DF_METHODS = ["predict_interval", "predict_quantiles", "predict_var"]


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
@pytest.mark.parametrize("mtype", PANEL_MTYPES)
def test_vectorization_series_to_panel_proba(method, mtype):
    """Test that forecaster vectorization works for Panel data, predict_proba.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    n_instances = 10

    y = _make_panel(n_instances=n_instances, random_state=42, return_mtype=mtype)

    est = ARIMA().fit(y)
    y_pred = getattr(est, method)([1, 2, 3])

    if method in ["predict_interval", "predict_quantiles"]:
        expected_mtype = method.replace("ict", "")
    elif method in ["predict_var"]:
        expected_mtype = "pd-multiindex"
    else:
        RuntimeError(f"bug in test, unreachable state, method {method} queried")

    valid, _, _ = check_is_mtype(y_pred, expected_mtype, return_metadata=True)

    msg = (
        f"vectorization of forecaster method {method} does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
@pytest.mark.parametrize("mtype", HIER_MTYPES)
def test_vectorization_series_to_hier_proba(method, mtype):
    """Test that forecaster vectorization works for Hierarchical data, predict_proba.

    This test passes Hierarchical data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    hierarchy_levels = (2, 4)
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)
    y = convert(y, from_type="pd_multiindex_hier", to_type=mtype)

    est = ARIMA().fit(y)
    y_pred = getattr(est, method)([1, 2, 3])

    if method in ["predict_interval", "predict_quantiles"]:
        expected_mtype = method.replace("ict", "")
    elif method in ["predict_var"]:
        expected_mtype = "pd_multiindex_hier"
    else:
        RuntimeError(f"bug in test, unreachable state, method {method} queried")

    valid, _, _ = check_is_mtype(y_pred, expected_mtype, return_metadata=True)

    msg = (
        f"vectorization of forecaster method {method} does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
def test_vectorization_preserves_row_index_names(method):
    """Test that forecaster vectorization preserves row index names in forecast."""
    hierarchy_levels = (2, 4)
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)

    est = ARIMA().fit(y, fh=[1, 2, 3])
    y_pred = getattr(est, method)()

    msg = (
        f"vectorization of forecaster method {method} changes row index names, "
        f"but it shouldn't. Tested using the ARIMA forecaster."
    )

    assert y_pred.index.names == y.index.names, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("mtype", HIER_MTYPES)
@pytest.mark.parametrize("exogeneous", [True, False])
def test_vectorization_multivariate(mtype, exogeneous):
    """Test that forecaster vectorization preserves row index names in forecast."""
    hierarchy_levels = (2, 4)
    n_instances = reduce(mul, hierarchy_levels)

    y = _make_hierarchical(
        hierarchy_levels=hierarchy_levels, random_state=84, n_columns=2
    )

    if exogeneous:
        y_fit = get_window(y, lag=pd.Timedelta("3D"))
        X_fit = y_fit
        X_pred = get_window(y, window_length=pd.Timedelta("3D"), lag=pd.Timedelta("0D"))
    else:
        y_fit = y
        X_fit = None
        X_pred = None

    est = ARIMA().fit(y=y_fit, X=X_fit, fh=[1, 2, 3])
    y_pred = est.predict(X=X_pred)
    valid, _, metadata = check_is_mtype(y_pred, mtype, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )
    assert valid, msg

    msg = (
        "vectorization over variables produces wrong set of variables in predict, "
        f"expected {y_fit.columns}, found {y_pred.columns}"
    )
    assert set(y_fit.columns) == set(y_pred.columns), msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dynamic_tags_reset_properly():
    """Test that dynamic tags are being reset properly."""
    from sktime.forecasting.compose import MultiplexForecaster
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.var import VAR

    # this forecaster will have the scitype:y tag set to "univariate"
    f = MultiplexForecaster([("foo", ThetaForecaster()), ("var", VAR())])
    f.set_params(selected_forecaster="var")

    X_multivariate = _make_series(n_columns=2)
    # fit should reset the estimator, and set scitype:y tag to "multivariate"
    # the fit will cause an error if this is not happening properly
    f.fit(X_multivariate)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_predict_residuals():
    """Test that predict_residuals has no side-effect."""
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.theta import ThetaForecaster

    y = _make_series(n_columns=1)
    y_train, y_test = temporal_train_test_split(y)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12)
    forecaster.fit(y_train, fh=fh)

    y_pred_1 = forecaster.predict()
    y_resid = forecaster.predict_residuals()
    y_pred_2 = forecaster.predict()
    assert_series_equal(y_pred_1, y_pred_2)
    assert y_resid.index.equals(y_train.index)
