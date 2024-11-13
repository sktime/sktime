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
from sktime.forecasting.compose import (
    ForecastingPipeline,
    TransformedTargetForecaster,
    YfromX,
    make_reduction,
)
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.split import ExpandingWindowSplitter, temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import OptionalPassthrough, YtoX
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.feature_selection import FeatureSelection
from sktime.transformations.series.fourier import FourierFeatures
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils._testing.series import _make_series
from sktime.utils.estimators import MockForecaster


@pytest.mark.skipif(
    not run_test_for_class([TransformedTargetForecaster, TabularToSeriesAdaptor]),
    reason="run test only if softdeps are present and incrementally (if requested)",
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


@pytest.mark.skipif(
    not run_test_for_class(TransformedTargetForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    not run_test_for_class(
        [
            AutoETS,
            TransformedTargetForecaster,
            OptionalPassthrough,
            ForecastingPipeline,
        ]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_nesting_pipelines():
    """Test that nesting of pipelines works."""
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


@pytest.mark.skipif(
    not run_test_for_class(TransformedTargetForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class(
        [
            TransformedTargetForecaster,
            OptionalPassthrough,
            ForecastingPipeline,
            TabularToSeriesAdaptor,
            make_reduction,
        ]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    not run_test_for_class([Aggregator, ForecastingPipeline]),
    reason="run test only if softdeps are present and incrementally (if requested)",
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

    # simple forecaster that can deal with exogenous data
    yfromx = YfromX.create_test_instance()

    # Aggregator creates indices for y (via *), then for X (via ForecastingPipeline)
    f = Aggregator() * ForecastingPipeline([Aggregator(), yfromx])

    f.fit(y=y_train, X=X_train, fh=1)
    y_pred = f.predict(X=X_test)

    # some basic expected output format checks
    assert isinstance(y_pred, pd.DataFrame)
    assert isinstance(y_pred.index, pd.MultiIndex)
    assert len(y_pred) == 9


@pytest.mark.skipif(
    not run_test_for_class([Aggregator, ForecastingPipeline]),
    reason="run test only if softdeps are present and incrementally (if requested)",
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

    # simple forecaster that can deal with exogenous data
    yfromx = YfromX.create_test_instance()

    # Aggregator creates indices for X (via ForecastingPipeline), then for y (via *)
    f = ForecastingPipeline([Aggregator(), Aggregator() * yfromx])

    f.fit(y=y_train, X=X_train, fh=1)
    y_pred = f.predict(X=X_test)

    # some basic expected output format checks
    assert isinstance(y_pred, pd.DataFrame)
    assert isinstance(y_pred.index, pd.MultiIndex)
    assert len(y_pred) == 9


@pytest.mark.skipif(
    not run_test_for_class([TabularToSeriesAdaptor, TransformedTargetForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    not run_test_for_class([TabularToSeriesAdaptor, ForecastingPipeline]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecasting_pipeline_dunder_exog():
    """Test forecasting pipeline dunder for exogeneous transformation."""
    y = _make_series()
    y_train, y_test = temporal_train_test_split(y)
    X = _make_series(n_columns=2)
    X_train, X_test = temporal_train_test_split(X)

    # simple forecaster that can deal with exogenous data
    yfromx = YfromX.create_test_instance()

    forecaster = ExponentTransformer() ** MinMaxScaler() ** yfromx
    forecaster_alt = (ExponentTransformer() * MinMaxScaler()) ** yfromx

    assert isinstance(forecaster, ForecastingPipeline)
    assert isinstance(forecaster.steps[0], ExponentTransformer)
    assert isinstance(forecaster.steps[1], TabularToSeriesAdaptor)
    assert isinstance(forecaster.steps[2], YfromX)
    assert isinstance(forecaster_alt, ForecastingPipeline)
    assert isinstance(forecaster_alt.steps[0], ExponentTransformer)
    assert isinstance(forecaster_alt.steps[1], TabularToSeriesAdaptor)
    assert isinstance(forecaster_alt.steps[2], YfromX)

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
        forecaster = YfromX.create_test_instance()
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


@pytest.mark.skipif(
    not run_test_for_class([TransformedTargetForecaster, ForecastingPipeline, Imputer]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tag_handles_missing_data():
    """Test missing data with Imputer in pipelines.

    Make sure that no exception is raised when NaN and Imputer is given. This test is
    based on bug issue #3547.
    """
    forecaster = MockForecaster()
    # make sure that test forecaster can't handle missing data
    forecaster.set_tags(**{"handles-missing-data": False})

    y = _make_series()
    y.iloc[10] = np.nan

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
    not run_test_for_class([TransformedTargetForecaster, ForecastingPipeline]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_subset_getitem():
    """Test subsetting using the [ ] dunder, __getitem__."""
    y = _make_series(n_columns=3)
    y.columns = ["x", "y", "z"]
    y_train, _ = temporal_train_test_split(y)
    X = _make_series(n_columns=3)
    X.columns = ["a", "b", "c"]
    X_train, X_test = temporal_train_test_split(X)

    # simple forecaster that can deal with exogenous data
    f = YfromX.create_test_instance()

    f_before = f[["a", "b"]]
    f_before_with_colon = f[["a", "b"], :]
    f_after_with_colon = f[:, ["x", "y"]]
    f_both = f[["a", "b"], ["y", "z"]]
    f_none = f[:, :]

    assert isinstance(f_before, ForecastingPipeline)
    assert isinstance(f_after_with_colon, TransformedTargetForecaster)
    assert isinstance(f_before_with_colon, ForecastingPipeline)
    assert isinstance(f_both, TransformedTargetForecaster)
    assert isinstance(f_none, YfromX)

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
    not run_test_for_class([YfromX, YtoX, Lag, Imputer]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_featurizer_forecastingpipeline_logic():
    """Test that ForecastingPipeline works with featurizer transformers without exog."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    # simple forecaster that can deal with exogenous data
    f = YfromX.create_test_instance()

    lagged_y_trafo = YtoX() * Lag(1, index_out="original") * Imputer()
    # we need to specify index_out="original" as otherwise ARIMA gets 1 and 2 ahead
    forecaster = lagged_y_trafo**f  # this uses lagged_y_trafo to generate X

    forecaster.fit(y_train, X=X_train, fh=[1])  # try to forecast next year
    forecaster.predict(X=X_test)  # dummy X to predict next year


@pytest.mark.skipif(
    not run_test_for_class([YfromX, FeatureSelection, TransformedTargetForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_exogenousx_ignore_tag_set():
    """Tests that TransformedTargetForecaster sets X tag for feature selection.

    If the forecaster ignores X, but the feature selector does not, then the
    ignores-exogeneous-X tag should be correctly set to False, not True.

    This is the failure case in bug report #5518.

    More generally, the tag should be set to True iff all steps in the pipeline
    ignore X.
    """
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


@pytest.mark.skipif(
    not run_test_for_class([YfromX, ForecastingPipeline, FeatureSelection]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pipeline_exogenous_none():
    """Test ForecastingPipeline works with a transformer returning None."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=3)

    # simple forecaster that can deal with exogenous data
    yfromx = YfromX.create_test_instance()

    pipe = ForecastingPipeline(
        [
            ("select_X", FeatureSelection(method="none")),
            ("yfromx", yfromx),
        ]
    )

    pipe.fit(y_train, X_train, fh=[1, 2, 3])
    y_pred = pipe.predict(X=X_test)
    assert np.all(y_pred.index == y_test.index)


@pytest.mark.skipif(
    not run_test_for_class([ForecastingPipeline, YfromX, FourierFeatures]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pipeline_featurizer_noexog():
    """Test that ForecastingPipeline works with featurizer transformers without exog

    Tests for failure case in #5975.
    Compared to test_pipeline_exogenous_none,
    this tests that transformers are executed properly even if X=None.
    """
    calls_per_min_low = 100 * 60
    calls_per_min_high = 500 * 60
    data = np.random.randint(low=calls_per_min_low, high=calls_per_min_high, size=100)
    calls = pd.Series(data)

    fh = range(1, 10)

    fcst = YfromX.create_test_instance()

    pipe = ForecastingPipeline(
        [
            YtoX(),
            FourierFeatures(sp_list=[24, 24 * 7], fourier_terms_list=[10, 5]),
            fcst,
        ]
    )

    y_pred = pipe.fit_predict(y=calls, fh=fh)

    # if the pipeline skips the FourierFeatures step,
    # then the predictions would be all constant, we test that this is not the case
    assert not np.allclose(y_pred.diff()[1:], np.zeros_like(y_pred[1:]))


@pytest.mark.skipif(
    not run_test_for_class(
        [ForecastingPipeline, TransformedTargetForecaster, YfromX, Detrender],
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pipeline_display():
    """Test that pipeline displays correctly."""
    from sktime.forecasting.compose import TransformedTargetForecaster, YfromX
    from sktime.transformations.series.detrend import Detrender

    f = TransformedTargetForecaster([Detrender(), YfromX.create_test_instance()])
    f._sk_visual_block_()

    f = ForecastingPipeline([Detrender(), YfromX.create_test_instance()])
    f._sk_visual_block_()
