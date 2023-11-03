#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for forecasting pipelines."""

__author__ = ["mloning", "fkiraly"]
__all__ = []

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from sktime.datasets import load_airline, load_longley
from sktime.datatypes import get_examples
from sktime.datatypes._utilities import get_window
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import (
    ForecastingPipeline,
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.split import ExpandingWindowSplitter, temporal_train_test_split
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils._testing.series import _make_series
from sktime.utils.estimators import MockForecaster
from sktime.utils.validation._dependencies import (
    _check_estimator_deps,
    _check_soft_dependencies,
)


def test_pipeline():
    """Test results of TransformedTargetForecaster."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    forecaster = TransformedTargetForecaster(
        [
            ("t1", ExponentTransformer()),
            ("t2", TabularToSeriesAdaptor(MinMaxScaler())),
            ("forecaster", NaiveForecaster()),
        ]
    )
    fh = np.arange(len(y_test)) + 1
    forecaster.fit(y_train, fh=fh)
    actual = forecaster.predict()

    def compute_expected_y_pred(y_train, fh):
        # fitting
        yt = y_train.copy()
        t1 = ExponentTransformer()
        yt = t1.fit_transform(yt)
        t2 = TabularToSeriesAdaptor(MinMaxScaler())
        yt = t2.fit_transform(yt)
        forecaster = NaiveForecaster()
        forecaster.fit(yt, fh=fh)

        # predicting
        y_pred = forecaster.predict()
        y_pred = t2.inverse_transform(y_pred)
        y_pred = t1.inverse_transform(y_pred)
        return y_pred

    expected = compute_expected_y_pred(y_train, fh)
    np.testing.assert_array_equal(actual, expected)


def test_skip_inverse_transform():
    """Test transformers with skip-inverse-transform tag in pipeline."""
    y = load_airline()
    # add nan and outlier
    y.iloc[3] = np.nan
    y.iloc[4] = y.iloc[4] * 20

    y_train, y_test = temporal_train_test_split(y)
    forecaster = TransformedTargetForecaster(
        [
            ("t1", HampelFilter(window_length=12)),
            ("t2", Imputer(method="mean")),
            ("forecaster", NaiveForecaster()),
        ]
    )
    fh = np.arange(len(y_test)) + 1
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict()
    assert isinstance(y_pred, pd.Series)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_nesting_pipelines():
    """Test that nesting of pipelines works."""
    from sktime.forecasting.ets import AutoETS
    from sktime.transformations.compose import OptionalPassthrough
    from sktime.transformations.series.boxcox import LogTransformer
    from sktime.transformations.series.detrend import Detrender
    from sktime.utils._testing.scenarios_forecasting import (
        ForecasterFitPredictUnivariateWithX,
    )

    pipe = ForecastingPipeline(
        steps=[
            ("logX", OptionalPassthrough(LogTransformer())),
            ("detrenderX", OptionalPassthrough(Detrender(forecaster=AutoETS()))),
            (
                "etsforecaster",
                TransformedTargetForecaster(
                    steps=[
                        ("log", OptionalPassthrough(LogTransformer())),
                        ("autoETS", AutoETS()),
                    ]
                ),
            ),
        ]
    )

    scenario = ForecasterFitPredictUnivariateWithX()

    scenario.run(pipe, method_sequence=["fit", "predict"])


def test_pipeline_with_detrender():
    """Tests a specific pipeline that triggers multiple back/forth conversions."""
    y = load_airline()

    trans_fc = TransformedTargetForecaster(
        [
            ("detrender", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
            ("forecaster", NaiveForecaster(strategy="last")),
        ]
    )
    trans_fc.fit(y)
    trans_fc.predict(1)


def test_pipeline_with_dimension_changing_transformer():
    """Example of pipeline with dimension changing transformer.

    The code below should run without generating any errors.  Issues can arise from
    using Differencer in the pipeline.
    """
    y, X = load_longley()

    # split train/test both y and X
    fh = [1, 2, 3]
    train_model, test_model = temporal_train_test_split(y, fh=fh)
    X_train = X[X.index.isin(train_model.index)]

    # pipeline
    pipe = TransformedTargetForecaster(
        steps=[
            ("log", OptionalPassthrough(LogTransformer())),
            ("differencer", Differencer(na_handling="drop_na")),
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "myforecasterpipe",
                ForecastingPipeline(
                    steps=[
                        ("logX", OptionalPassthrough(LogTransformer())),
                        ("differencerX", Differencer(na_handling="drop_na")),
                        ("scalerX", TabularToSeriesAdaptor(StandardScaler())),
                        ("myforecaster", make_reduction(SVR())),
                    ]
                ),
            ),
        ]
    )

    # cv setup
    N_cv_fold = 1
    step_cv = 1
    cv = ExpandingWindowSplitter(
        initial_window=len(train_model) - (N_cv_fold - 1) * step_cv - len(fh),
        step_length=step_cv,
        fh=fh,
    )

    param_grid = [
        {
            "log__passthrough": [False],
            "myforecasterpipe__logX__passthrough": [False],
            "myforecasterpipe__myforecaster__window_length": [2, 3],
            "myforecasterpipe__myforecaster__estimator__C": [10, 100],
        },
        {
            "log__passthrough": [True],
            "myforecasterpipe__logX__passthrough": [True],
            "myforecasterpipe__myforecaster__window_length": [2, 3],
            "myforecasterpipe__myforecaster__estimator__C": [10, 100],
        },
    ]

    # grid search
    gscv = ForecastingGridSearchCV(
        forecaster=pipe, cv=cv, param_grid=param_grid, verbose=1
    )

    # fit
    gscv.fit(train_model, X=X_train)


@pytest.mark.skipif(
    not _check_estimator_deps(SARIMAX, severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_nested_pipeline_with_index_creation_y_before_X():
    """Tests a nested pipeline where y indices are created before X indices.

    The potential failure mode is the pipeline failing as y has more indices than X, in
    an intermediate stage and erroneous checks from the pipeline raise an error.
    """
    X = get_examples("pd_multiindex_hier")[0]
    y = get_examples("pd_multiindex_hier")[1]

    X_train = get_window(X, lag=1)
    y_train = get_window(y, lag=1)
    X_test = get_window(X, window_length=1)

    # Aggregator creates indices for y (via *), then for X (via ForecastingPipeline)
    f = Aggregator() * ForecastingPipeline([Aggregator(), SARIMAX()])

    f.fit(y=y_train, X=X_train, fh=1)
    y_pred = f.predict(X=X_test)

    # some basic expected output format checks
    assert isinstance(y_pred, pd.DataFrame)
    assert isinstance(y_pred.index, pd.MultiIndex)
    assert len(y_pred) == 9


@pytest.mark.skipif(
    not _check_estimator_deps(SARIMAX, severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_nested_pipeline_with_index_creation_X_before_y():
    """Tests a nested pipeline where X indices are created before y indices.

    The potential failure mode is the pipeline failing as X has more indices than y, in
    an intermediate stage and erroneous checks from the pipeline raise an error.
    """
    X = get_examples("pd_multiindex_hier")[0]
    y = get_examples("pd_multiindex_hier")[1]

    X_train = get_window(X, lag=1)
    y_train = get_window(y, lag=1)
    X_test = get_window(X, window_length=1)

    # Aggregator creates indices for X (via ForecastingPipeline), then for y (via *)
    f = ForecastingPipeline([Aggregator(), Aggregator() * SARIMAX()])

    f.fit(y=y_train, X=X_train, fh=1)
    y_pred = f.predict(X=X_test)

    # some basic expected output format checks
    assert isinstance(y_pred, pd.DataFrame)
    assert isinstance(y_pred.index, pd.MultiIndex)
    assert len(y_pred) == 9


def test_forecasting_pipeline_dunder_endog():
    """Test forecasting pipeline dunder for endogeneous transformation."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    forecaster = ExponentTransformer() * MinMaxScaler() * NaiveForecaster()

    assert isinstance(forecaster, TransformedTargetForecaster)
    assert isinstance(forecaster.steps[0], ExponentTransformer)
    assert isinstance(forecaster.steps[1], TabularToSeriesAdaptor)
    assert isinstance(forecaster.steps[2], NaiveForecaster)

    fh = np.arange(len(y_test)) + 1
    forecaster.fit(y_train, fh=fh)
    actual = forecaster.predict()

    def compute_expected_y_pred(y_train, fh):
        # fitting
        yt = y_train.copy()
        t1 = ExponentTransformer()
        yt = t1.fit_transform(yt)
        t2 = TabularToSeriesAdaptor(MinMaxScaler())
        yt = t2.fit_transform(yt)
        forecaster = NaiveForecaster()
        forecaster.fit(yt, fh=fh)

        # predicting
        y_pred = forecaster.predict()
        y_pred = t2.inverse_transform(y_pred)
        y_pred = t1.inverse_transform(y_pred)
        return y_pred

    expected = compute_expected_y_pred(y_train, fh)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _check_estimator_deps(SARIMAX, severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_forecasting_pipeline_dunder_exog():
    """Test forecasting pipeline dunder for exogeneous transformation."""
    y = _make_series()
    y_train, y_test = temporal_train_test_split(y)
    X = _make_series(n_columns=2)
    X_train, X_test = temporal_train_test_split(X)

    forecaster = ExponentTransformer() ** MinMaxScaler() ** SARIMAX(random_state=3)
    forecaster_alt = (ExponentTransformer() * MinMaxScaler()) ** SARIMAX(random_state=3)

    assert isinstance(forecaster, ForecastingPipeline)
    assert isinstance(forecaster.steps[0], ExponentTransformer)
    assert isinstance(forecaster.steps[1], TabularToSeriesAdaptor)
    assert isinstance(forecaster.steps[2], SARIMAX)
    assert isinstance(forecaster_alt, ForecastingPipeline)
    assert isinstance(forecaster_alt.steps[0], ExponentTransformer)
    assert isinstance(forecaster_alt.steps[1], TabularToSeriesAdaptor)
    assert isinstance(forecaster_alt.steps[2], SARIMAX)

    fh = np.arange(len(y_test)) + 1
    forecaster.fit(y_train, fh=fh, X=X_train)
    actual = forecaster.predict(X=X_test)

    forecaster_alt.fit(y_train, fh=fh, X=X_train)
    actual_alt = forecaster_alt.predict(X=X_test)

    def compute_expected_y_pred(y_train, X_train, X_test, fh):
        # fitting
        yt = y_train.copy()
        Xt = X_train.copy()
        t1 = ExponentTransformer()
        Xt = t1.fit_transform(Xt)
        t2 = TabularToSeriesAdaptor(MinMaxScaler())
        Xt = t2.fit_transform(Xt)
        forecaster = SARIMAX(random_state=3)
        forecaster.fit(yt, fh=fh, X=Xt)

        # predicting
        Xtt = X_test.copy()
        Xtt = t1.transform(Xtt)
        Xtt = t2.transform(Xtt)
        y_pred = forecaster.predict(X=Xtt)
        return y_pred

    expected = compute_expected_y_pred(y_train, X_train, X_test, fh)
    _assert_array_almost_equal(actual, expected, decimal=2)
    _assert_array_almost_equal(actual_alt, expected, decimal=2)


def test_tag_handles_missing_data():
    """Test missing data with Imputer in pipelines.

    Make sure that no exception is raised when NaN and Imputer is given. This test is
    based on bug issue #3547.
    """
    forecaster = MockForecaster()
    # make sure that test forecaster can't handle missing data
    forecaster.set_tags(**{"handles-missing-data": False})

    y = _make_series()
    y[10] = np.nan

    # test only TransformedTargetForecaster
    y_pipe = TransformedTargetForecaster(
        steps=[("transformer_y", Imputer()), ("model", forecaster)]
    )
    y_pipe.fit(y)

    # test TransformedTargetForecaster and ForecastingPipeline nested
    y_pipe = TransformedTargetForecaster(
        steps=[("transformer_y", Imputer()), ("model", forecaster)]
    )
    X_pipe = ForecastingPipeline(steps=[("forecaster", y_pipe)])
    X_pipe.fit(y)


@pytest.mark.skipif(
    not _check_estimator_deps(SARIMAX, severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_subset_getitem():
    """Test subsetting using the [ ] dunder, __getitem__."""
    y = _make_series(n_columns=3)
    y.columns = ["x", "y", "z"]
    y_train, _ = temporal_train_test_split(y)
    X = _make_series(n_columns=3)
    X.columns = ["a", "b", "c"]
    X_train, X_test = temporal_train_test_split(X)

    f = SARIMAX(random_state=3)

    f_before = f[["a", "b"]]
    f_before_with_colon = f[["a", "b"], :]
    f_after_with_colon = f[:, ["x", "y"]]
    f_both = f[["a", "b"], ["y", "z"]]
    f_none = f[:, :]

    assert isinstance(f_before, ForecastingPipeline)
    assert isinstance(f_after_with_colon, TransformedTargetForecaster)
    assert isinstance(f_before_with_colon, ForecastingPipeline)
    assert isinstance(f_both, TransformedTargetForecaster)
    assert isinstance(f_none, SARIMAX)

    y_pred = f.fit(y_train, X_train, fh=X_test.index).predict(X=X_test)

    y_pred_f_before = f_before.fit(y_train, X_train, fh=X_test.index).predict(X=X_test)
    y_pred_f_before_with_colon = f_before_with_colon.fit(
        y_train, X_train, fh=X_test.index
    ).predict(X=X_test)
    y_pred_f_after_with_colon = f_after_with_colon.fit(
        y_train, X_train, fh=X_test.index
    ).predict(X=X_test)
    y_pred_f_both = f_both.fit(y_train, X_train, fh=X_test.index).predict(X=X_test)
    y_pred_f_none = f_none.fit(y_train, X_train, fh=X_test.index).predict(X=X_test)

    _assert_array_almost_equal(y_pred, y_pred_f_none)
    _assert_array_almost_equal(y_pred_f_before, y_pred_f_before_with_colon)
    _assert_array_almost_equal(y_pred_f_before, y_pred_f_both[["y", "z"]])
    _assert_array_almost_equal(y_pred_f_after_with_colon, y_pred_f_none[["x", "y"]])
    _assert_array_almost_equal(y_pred_f_before_with_colon, y_pred_f_both[["y", "z"]])


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_forecastx_logic():
    """Test that ForecastX logic is as expected, compared to manual execution."""
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.compose import ForecastX
    from sktime.forecasting.var import VAR
    from sktime.split import temporal_train_test_split

    # test case: using pipeline execution
    y, X = load_longley()
    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=3)
    fh = ForecastingHorizon([1, 2, 3])
    columns = ["ARMED", "POP"]

    # ForecastX
    pipe = ForecastX(
        forecaster_X=VAR(),
        forecaster_y=SARIMAX(),
        columns=columns,
    )
    pipe = pipe.fit(y_train, X=X_train, fh=fh)
    # dropping ["ARMED", "POP"] = columns where we expect not to have future values
    y_pred = pipe.predict(fh=fh, X=X_test.drop(columns=columns))

    # comparison case: manual execution
    # fit y forecaster
    arima = SARIMAX().fit(y_train, X=X_train)

    # fit and predict X forecaster
    var = VAR()
    var.fit(X_train[columns])
    var_pred = var.predict(fh)

    # predict y forecaster with predictions from VAR
    X_pred = pd.concat([X_test.drop(columns=columns), var_pred], axis=1)
    y_pred_manual = arima.predict(fh=fh, X=X_pred)

    # compare that test and comparison case results are equal
    assert np.allclose(y_pred, y_pred_manual)


@pytest.mark.skipif(
    not _check_estimator_deps(ARIMA, severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_forecastx_fit_behavior():
    from sktime.forecasting.compose import ForecastX
    from sktime.forecasting.model_selection import temporal_train_test_split

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    pipe = ForecastX(
        forecaster_X=NaiveForecaster(),
        forecaster_y=ARIMA(),
    )
    pipe = pipe.fit(y_train, X=X_train, fh=y_test.index)
    y_pred_forecast_X_use_gt = pipe.predict(fh=y_test.index)

    naive = NaiveForecaster()
    naive.fit(X_train)
    x_pred_train = naive.predict(fh=X_train.index)
    arima = ARIMA()
    arima.fit(y_train, X_train)

    y_pred = arima.predict(fh=y_test.index, X=naive.predict(fh=y_test.index))

    pd.testing.assert_series_equal(y_pred_forecast_X_use_gt, y_pred)

    pipe = ForecastX(
        forecaster_X=NaiveForecaster(),
        forecaster_y=ARIMA(),
        fit_behaviour="use_forecast",
    )
    pipe = pipe.fit(y_train, X=X_train, fh=y_test.index)
    y_pred_forecast_X_use_forecast = pipe.predict(fh=y_test.index)

    arima = ARIMA()
    arima.fit(y_train, x_pred_train)
    y_pred = arima.predict(fh=y_test.index, X=naive.predict(fh=y_test.index))

    pd.testing.assert_series_equal(y_pred_forecast_X_use_forecast, y_pred)


def test_forecastx_attrib_broadcast():
    """Test ForecastX broadcasting and forecaster attributes."""
    from sktime.forecasting.compose import ForecastX
    from sktime.forecasting.naive import NaiveForecaster

    df = pd.DataFrame(
        {
            "a": ["series_1", "series_1", "series_1"],
            "b": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "c": [1, 2, 3],
            "d": [4, 5, 6],
            "e": [7, 8, 9],
        }
    )
    df = df.set_index(["a", "b"])

    model = ForecastX(NaiveForecaster(), NaiveForecaster())

    model_1 = model.clone()
    model_1.fit(df[["c"]], X=df[["d", "e"]], fh=[1, 2, 3])

    assert not hasattr(model_1, "forecaster_X_")

    assert hasattr(model_1, "forecaster_y_")
    assert isinstance(model_1.forecaster_y_, NaiveForecaster)
    assert model_1.forecaster_y_.is_fitted

    model_2 = model.clone()
    model_2.fit(df[["c", "d"]], X=df[["e"]], fh=[1, 2, 3])

    assert not hasattr(model_2, "forecaster_X_")

    assert hasattr(model_2, "forecaster_y_")
    assert isinstance(model_2.forecaster_y_, NaiveForecaster)
    assert model_2.forecaster_y_.is_fitted


def test_forecastx_skip_forecaster_X_fitting_logic():
    """Test that ForecastX does not fit forecaster_X, if forecaster_y ignores X"""
    from sklearn.linear_model import LinearRegression

    from sktime.forecasting.compose import ForecastX, YfromX

    y, X = load_longley()

    fh = [1, 2, 3]

    model_supporting_exogenous = YfromX(LinearRegression())
    model_ignoring_exogenous = NaiveForecaster()

    model_1 = ForecastX(
        model_supporting_exogenous.clone(), model_supporting_exogenous.clone()
    )
    model_2 = ForecastX(
        model_supporting_exogenous.clone(), model_ignoring_exogenous.clone()
    )
    model_3 = ForecastX(
        model_ignoring_exogenous.clone(), model_supporting_exogenous.clone()
    )
    model_4 = ForecastX(
        model_ignoring_exogenous.clone(), model_ignoring_exogenous.clone()
    )

    assert hasattr(model_1, "forecaster_y")
    assert hasattr(model_2, "forecaster_y")
    assert hasattr(model_3, "forecaster_y")
    assert hasattr(model_4, "forecaster_y")

    assert hasattr(model_1, "forecaster_X")
    assert hasattr(model_2, "forecaster_X")
    assert hasattr(model_3, "forecaster_X")
    assert hasattr(model_4, "forecaster_X")

    assert not hasattr(model_1, "forecaster_y_")
    assert not hasattr(model_2, "forecaster_y_")
    assert not hasattr(model_3, "forecaster_y_")
    assert not hasattr(model_4, "forecaster_y_")

    assert not hasattr(model_1, "forecaster_X_")
    assert not hasattr(model_2, "forecaster_X_")
    assert not hasattr(model_3, "forecaster_X_")
    assert not hasattr(model_4, "forecaster_X_")

    model_1.fit(y, X=X, fh=fh)
    model_2.fit(y, X=X, fh=fh)
    model_3.fit(y, X=X, fh=fh)
    model_4.fit(y, X=X, fh=fh)

    assert hasattr(model_1, "forecaster_y_")
    assert hasattr(model_2, "forecaster_y_")
    assert hasattr(model_3, "forecaster_y_")
    assert hasattr(model_4, "forecaster_y_")

    assert model_1.forecaster_y_.is_fitted
    assert model_2.forecaster_y_.is_fitted
    assert model_3.forecaster_y_.is_fitted
    assert model_4.forecaster_y_.is_fitted

    assert hasattr(model_1, "forecaster_X_")
    assert hasattr(model_2, "forecaster_X_")
    assert not hasattr(model_3, "forecaster_X_")
    assert not hasattr(model_4, "forecaster_X_")

    assert model_1.forecaster_X_.is_fitted
    assert model_2.forecaster_X_.is_fitted


@pytest.mark.parametrize(
    "forecasting_algorithm", [make_reduction(SVR(), window_length=2), NaiveForecaster()]
)
@pytest.mark.parametrize(
    "future_unknown_columns",
    [["GNPDEFL", "GNP"], ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP"], None],
)
def test_forecastx_flow_known_unknown_columns(
    forecasting_algorithm, future_unknown_columns
):
    """Test that ForecastX does not fit forecaster_X, if forecaster_y ignores X"""
    from sktime.forecasting.compose import ForecastX

    y, X = load_longley()

    fh = [1, 2]

    y_train_val, y_test, X_train_val, X_test = temporal_train_test_split(
        y, X, test_size=max(fh)
    )
    y_train, y_val, X_train, X_val = temporal_train_test_split(
        y_train_val, X_train_val, test_size=max(fh)
    )

    model = ForecastX(
        forecasting_algorithm.clone(),
        forecasting_algorithm.clone(),
        columns=future_unknown_columns,
    )

    assert hasattr(model, "forecaster_y")
    assert hasattr(model, "forecaster_X")

    assert not hasattr(model, "forecaster_y_")
    assert not hasattr(model, "forecaster_X_")

    model.fit(y_train, X=X_train, fh=fh)

    assert hasattr(model, "forecaster_y_")
    assert model.forecaster_y_.is_fitted

    if model.get_tag("ignores-exogeneous-X"):
        assert not hasattr(model, "forecaster_X_")
    else:
        assert hasattr(model, "forecaster_X_")
        assert model.forecaster_X_.is_fitted

    y_val_pred = model.predict(X=X_test)
    np.testing.assert_array_equal(y_val.index, y_val_pred.index)

    model.update(y_val, X=X_val)

    y_test_pred = model.predict(X=X_test)
    np.testing.assert_array_equal(y_test.index, y_test_pred.index)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_featurizer_forecastingpipeline_logic():
    """Test that ForecastingPipeline works with featurizer transformers without exog."""
    from sktime.forecasting.sarimax import SARIMAX
    from sktime.transformations.compose import YtoX
    from sktime.transformations.series.impute import Imputer
    from sktime.transformations.series.lag import Lag

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    lagged_y_trafo = YtoX() * Lag(1, index_out="original") * Imputer()
    # we need to specify index_out="original" as otherwise ARIMA gets 1 and 2 ahead
    forecaster = lagged_y_trafo ** SARIMAX()  # this uses lagged_y_trafo to generate X

    forecaster.fit(y_train, X=X_train, fh=[1])  # try to forecast next year
    forecaster.predict(X=X_test)  # dummy X to predict next year


def test_exogenousx_ignore_tag_set():
    """Tests that TransformedTargetForecaster sets X tag for feature selection.

    If the forecaster ignores X, but the feature selector does not, then the
    ignores-exogeneous-X tag should be correctly set to False, not True.

    This is the failure case in bug report #5518.

    More generally, the tag should be set to True iff all steps in the pipeline
    ignore X.
    """
    from sktime.forecasting.compose import YfromX
    from sktime.transformations.series.feature_selection import FeatureSelection

    fcst_does_not_ignore_x = YfromX.create_test_instance()
    fcst_ignores_x = NaiveForecaster()

    trafo_ignores_x = ExponentTransformer()
    trafo_does_not_ignore_x = FeatureSelection()

    # check that ignores-exogeneous-X tag is set correctly
    pipe1 = trafo_ignores_x * fcst_does_not_ignore_x
    pipe2 = trafo_ignores_x * fcst_ignores_x
    pipe3 = trafo_does_not_ignore_x * fcst_does_not_ignore_x
    pipe4 = trafo_does_not_ignore_x * fcst_ignores_x
    pipe5 = trafo_ignores_x * trafo_does_not_ignore_x * fcst_does_not_ignore_x
    pipe6 = trafo_ignores_x * trafo_does_not_ignore_x * fcst_ignores_x
    pipe7 = trafo_ignores_x * trafo_ignores_x * fcst_does_not_ignore_x
    pipe8 = trafo_ignores_x * fcst_ignores_x * trafo_does_not_ignore_x
    pipe9 = trafo_does_not_ignore_x * fcst_ignores_x * trafo_ignores_x
    pipe10 = trafo_ignores_x * fcst_ignores_x * trafo_ignores_x

    assert not pipe1.get_tag("ignores-exogeneous-X")
    assert pipe2.get_tag("ignores-exogeneous-X")
    assert not pipe3.get_tag("ignores-exogeneous-X")
    assert not pipe4.get_tag("ignores-exogeneous-X")
    assert not pipe5.get_tag("ignores-exogeneous-X")
    assert not pipe6.get_tag("ignores-exogeneous-X")
    assert not pipe7.get_tag("ignores-exogeneous-X")
    assert not pipe8.get_tag("ignores-exogeneous-X")
    assert not pipe9.get_tag("ignores-exogeneous-X")
    assert pipe10.get_tag("ignores-exogeneous-X")
