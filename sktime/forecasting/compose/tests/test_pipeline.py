#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for forecasting pipelines."""

__author__ = ["mloning", "fkiraly"]
__all__ = []

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sktime.datasets import load_airline
from sktime.datatypes import get_examples
from sktime.datatypes._utilities import get_window
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import ForecastingPipeline, TransformedTargetForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter


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
    f = Aggregator() * ForecastingPipeline([Aggregator(), ARIMA()])

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
    f = ForecastingPipeline([Aggregator(), Aggregator() * ARIMA()])

    f.fit(y=y_train, X=X_train, fh=1)
    y_pred = f.predict(X=X_test)

    # some basic expected output format checks
    assert isinstance(y_pred, pd.DataFrame)
    assert isinstance(y_pred.index, pd.MultiIndex)
    assert len(y_pred) == 9
