"""Testing advanced functionality of the base class."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly", "ciaran-g"]

from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from sktime.datatypes import check_is_mtype, convert
from sktime.datatypes._utilities import get_cutoff, get_window
from sktime.forecasting.compose import YfromX
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.var import VAR
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.panel import _make_panel
from sktime.utils._testing.series import _make_series
from sktime.utils.dependencies import _check_estimator_deps, _check_soft_dependencies
from sktime.utils.parallel import _get_parallel_test_fixtures

PANEL_MTYPES = ["pd-multiindex", "nested_univ", "numpy3D"]
HIER_MTYPES = ["pd_multiindex_hier"]

# list of parallelization backends to test
BACKENDS = _get_parallel_test_fixtures("config")


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_soft_dependencies("skpro", severity="none"),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mtype", PANEL_MTYPES)
def test_vectorization_series_to_panel(mtype, backend):
    """Test that forecaster vectorization works for Panel data.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    n_instances = 10

    y = _make_panel(n_instances=n_instances, random_state=42, return_mtype=mtype)

    f = YfromX.create_test_instance()
    f.set_config(**backend.copy())
    y_pred = f.fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(
        y_pred, mtype, return_metadata=True, msg_return_dict="list"
    )

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the YfromX forecaster"
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
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mtype", HIER_MTYPES)
def test_vectorization_series_to_hier(mtype, backend):
    """Test that forecaster vectorization works for Hierarchical data.

    This test passes Hierarchical data to the YfromX forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    hierarchy_levels = (2, 4)
    n_instances = reduce(mul, hierarchy_levels)

    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)
    y = convert(y, from_type="pd_multiindex_hier", to_type=mtype)

    f = YfromX.create_test_instance()
    assert f.get_tags()["scitype:y"] == "univariate"  # check the assumption

    f.set_config(**backend.copy())
    y_pred = f.fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(
        y_pred, mtype, return_metadata=True, msg_return_dict="list"
    )

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the YfromX forecaster"
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
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_soft_dependencies("skpro", severity="none"),
    reason="run only if base module has changed or datatypes module has changed",
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

    est = _get_exog_proba_fcst()
    est.fit(y)
    y_pred = getattr(est, method)([1, 2, 3])

    if method in ["predict_interval", "predict_quantiles"]:
        expected_mtype = method.replace("ict", "")
    elif method in ["predict_var"]:
        expected_mtype = "pd-multiindex"
    else:
        RuntimeError(f"bug in test, unreachable state, method {method} queried")

    valid, _, _ = check_is_mtype(
        y_pred, expected_mtype, return_metadata=True, msg_return_dict="list"
    )

    msg = (
        f"vectorization of forecaster method {method} does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_soft_dependencies("skpro", severity="none"),
    reason="run only if base module has changed or datatypes module has changed",
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

    est = _get_exog_proba_fcst()
    est.fit(y)
    y_pred = getattr(est, method)([1, 2, 3])

    if method in ["predict_interval", "predict_quantiles"]:
        expected_mtype = method.replace("ict", "")
    elif method in ["predict_var"]:
        expected_mtype = "pd_multiindex_hier"
    else:
        RuntimeError(f"bug in test, unreachable state, method {method} queried")

    valid, _, _ = check_is_mtype(
        y_pred, expected_mtype, return_metadata=True, msg_return_dict="list"
    )

    msg = (
        f"vectorization of forecaster method {method} does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )

    assert valid, msg


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"])
    or not _check_soft_dependencies("skpro", severity="none"),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
def test_vectorization_preserves_row_index_names(method):
    """Test that forecaster vectorization preserves row index names in forecast."""
    hierarchy_levels = (2, 4)
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)

    est = _get_exog_proba_fcst()
    est.fit(y, fh=[1, 2, 3])
    y_pred = getattr(est, method)()

    msg = (
        f"vectorization of forecaster method {method} changes row index names, "
        f"but it shouldn't. Tested using the ARIMA forecaster."
    )

    assert y_pred.index.names == y.index.names, msg


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
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

    est = YfromX.create_test_instance()
    est.fit(y=y_fit, X=X_fit, fh=[1, 2, 3])
    y_pred = est.predict(X=X_pred)
    valid, _, metadata = check_is_mtype(
        y_pred, mtype, return_metadata=True, msg_return_dict="list"
    )

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
    not run_test_module_changed("sktime.forecasting.base"),
    reason="run only if base module has changed",
)
def test_col_vectorization_correct_col_order():
    """Test that forecaster vectorization preserves column index ordering.

    Failure case is as in issue #4683 where the column index is correct,
    but the values are in fact coming from forecasters in jumbled order.
    """
    cols = ["realgdp", "realcons", "realinv", "realgovt", "realdpi", "cpi", "m1"]
    vals = np.random.rand(5, 7)
    y = pd.DataFrame(vals, columns=cols)

    f = NaiveForecaster()
    # force univariate tag to trigger vectorization over columns for sure
    f.set_tags(**{"scitype:y": "univariate"})

    f.fit(y=y, fh=[1])
    y_pred = f.predict()

    # last value, so entries of last y column and y_pred should all be exactly equal
    # if they were jumbled, as in #4683 by lexicographic column name order,
    # this assertion would fail since the values are all different
    assert (y_pred == y.iloc[4]).all().all()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.base"),
    reason="run only if base module has changed",
)
def test_row_vectorization_correct_row_order():
    """Test that forecaster vectorization preserves row index ordering.

    Failure case is as in issue #5108 where the row index is correct,
    but the values are in fact coming from forecasters in jumbled order.
    """
    n_instances = 3
    n_points = 5

    t_ix = pd.date_range(start="2022-07-01", periods=n_points * n_instances, freq="D")
    y = pd.DataFrame(
        {
            "y": [i for i in range(n_points * n_instances)],
            "id": ["T1"] * n_points + ["T2"] * n_points + ["T11"] * n_points,
            "timestamp": t_ix,
        }
    ).set_index(["id", "timestamp"])

    fh = [1]

    forecaster = NaiveForecaster(strategy="last")

    forecaster.fit(y)
    y_pred = forecaster.predict(fh)

    last_ix = range(n_points - 1, n_points * n_instances, n_points)
    y_last = y.iloc[last_ix]

    assert all(y_last.index.get_level_values(0) == y_pred.index.get_level_values(0))
    assert all(y_last.values == y_pred.values)


@pytest.mark.skipif(
    not _check_estimator_deps([ThetaForecaster, VAR], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dynamic_tags_reset_properly():
    """Test that dynamic tags are being reset properly."""
    from sktime.forecasting.compose import MultiplexForecaster

    # this forecaster will have the scitype:y tag set to "univariate"
    f = MultiplexForecaster([("foo", ThetaForecaster()), ("var", VAR())])
    f.set_params(selected_forecaster="var")

    X_multivariate = _make_series(n_columns=2)
    # fit should reset the estimator, and set scitype:y tag to "multivariate"
    # the fit will cause an error if this is not happening properly
    f.fit(X_multivariate)


@pytest.mark.skipif(
    not _check_estimator_deps(ThetaForecaster, severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_predict_residuals():
    """Test that predict_residuals has no side-effect."""
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.split import temporal_train_test_split

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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
@pytest.mark.parametrize("nullable_type", ["Int64", "Float64", "boolean"])
def test_nullable_dtypes(nullable_type):
    """Test that basic forecasting vignette works with nullable DataFrame dtypes."""
    dtype = nullable_type

    X = pd.DataFrame()
    X["ints"] = pd.Series([1, 0] * 40, dtype=dtype)
    X.index = pd.date_range("1/1/21", periods=80)
    X_train = X.iloc[0:40]
    X_test = X.iloc[40:80]
    y = pd.Series([1, 0] * 20, dtype=dtype)
    y.index = pd.date_range("1/1/21", periods=40)

    f = YfromX.create_test_instance()

    fh = list(range(1, len(X_test) + 1))
    f.fit(X=X_train, y=y, fh=fh)
    y_pred = f.predict(X=X_test)
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == 40
    assert y_pred.dtype == "float64"


@pytest.mark.skipif(
    not _check_estimator_deps(VAR, severity="none")
    or not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_range_fh_in_fit():
    """Test using ``range`` in ``fit``."""
    test_dataset = _make_panel(n_instances=10, n_columns=5)

    var_model = VAR().fit(test_dataset, fh=range(1, 2 + 1))
    var_predictions = var_model.predict()

    assert isinstance(var_predictions, pd.DataFrame)
    assert var_predictions.shape == (10 * 2, 5)


@pytest.mark.skipif(
    not _check_estimator_deps(VAR, severity="none")
    or not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_range_fh_in_predict():
    """Test using ``range`` in ``predict``."""
    test_dataset = _make_panel(n_instances=10, n_columns=5)

    var_model = VAR().fit(test_dataset)

    with pytest.raises(
        ValueError,
        match=(
            "The forecasting horizon `fh` must be passed either to `fit` or `predict`,"
            " but was found in neither."
        ),
    ):
        _ = var_model.predict()

    var_predictions = var_model.predict(fh=range(1, 2 + 1))

    assert isinstance(var_predictions, pd.DataFrame)
    assert var_predictions.shape == (10 * 2, 5)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_remember_data():
    """Test that the ``remember_data`` flag works as expected."""
    from sktime.datasets import load_airline

    y = load_airline()
    X = load_airline()
    f = YfromX.create_test_instance()

    # turn off remembering _X, _y by config
    f.set_config(**{"remember_data": False})
    f.fit(y, X, fh=[1, 2, 3])

    assert f._X is None
    assert f._y is None

    f.set_config(**{"remember_data": True})
    f.fit(y, X, fh=[1, 2, 3])

    assert f._X is not None
    assert f._y is not None


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.forecasting.base", "sktime.datatypes"]),
    reason="run only if base module has changed or datatypes module has changed",
)
def test_panel_with_inner_freq():
    """Test that panel data with inner frequency set returns the correct predictions."""
    from sktime.datasets import load_airline

    y = load_airline()
    ind = pd.date_range(
        start="1960-01-01", periods=len(y.index), freq="H", name="datetime"
    )
    y = pd.DataFrame(y.values, index=ind, columns=["passengers"])

    y_pan = y.set_index([y.index.hour.rename("hour"), y.index]).sort_index()
    assert y_pan.loc[0].index.freq == pd.Timedelta("24H"), "Expected 24H frequency"

    fh = [1, 2]
    y_train, y_test = temporal_train_test_split(y_pan, test_size=len(fh))

    # fit update predict
    forecaster = NaiveForecaster()
    forecaster.fit(y_train)
    forecaster.update(y_test)
    y_pred_update = forecaster.predict(fh=fh)

    # fit no update
    forecaster.fit(y_pan)
    y_pred = forecaster.predict(fh=fh)
    assert y_pred.equals(y_pred_update), "Expected same predictions after update."

    # test predictions against no panel case (simple here :))
    forecaster = NaiveForecaster(sp=24)
    forecaster.fit(y)
    fh = np.arange(1, 49)
    y_pred_simple = forecaster.predict(fh=fh)

    msg = "Panel not returning same predictions as simple case."
    assert y_pred.droplevel("hour").sort_index().equals(y_pred_simple), msg


def _get_exog_proba_fcst():
    """Fast forecaster that can use exogenous data and make proba forecasts."""
    from sklearn.linear_model import LinearRegression
    from skpro.regression.residual import ResidualDouble

    lin_reg = LinearRegression()
    reg_proba = ResidualDouble(lin_reg, lin_reg)

    return YfromX(reg_proba)
