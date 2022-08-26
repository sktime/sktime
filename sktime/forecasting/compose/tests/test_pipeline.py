#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for forecasting pipelines."""

__author__ = ["mloning", "fkiraly"]
__all__ = []

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from sktime.datasets import load_airline, load_longley
from sktime.datatypes import get_examples
from sktime.datatypes._utilities import get_window
from sktime.forecasting.compose import (
    ForecastingPipeline,
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.compose import OptionalPassthrough
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils._testing.series import _make_series


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


def test_nesting_pipelines():
    """Test that nesting of pipelines works."""
    from sktime.forecasting.ets import AutoETS
    from sktime.transformations.series.boxcox import LogTransformer
    from sktime.transformations.series.compose import OptionalPassthrough
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

    The code below should run without generating any errors.  Issues
    can arise from using Differencer in the pipeline.
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
        start_with_window=True,
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


def test_nested_pipeline_with_index_creation_y_before_X():
    """Tests a nested pipeline where y indices are created before X indices.

    The potential failure mode is the pipeline failing as y has more indices than X,
    in an intermediate stage and erroneous checks from the pipeline raise an error.
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


def test_nested_pipeline_with_index_creation_X_before_y():
    """Tests a nested pipeline where X indices are created before y indices.

    The potential failure mode is the pipeline failing as X has more indices than y,
    in an intermediate stage and erroneous checks from the pipeline raise an error.
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
