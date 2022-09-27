#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Differencer functionality."""

__author__ = ["RNKuhns", "fkiraly", "ilkersigirci"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.transformations.series.difference import Differencer
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils.validation._dependencies import _check_soft_dependencies

y_airline = load_airline()
y_airline_df = pd.concat([y_airline, y_airline], axis=1)
y_airline_df.columns = ["Passengers 1", "Passengers 2"]

test_cases = [y_airline, y_airline_df]
lags_to_test = [1, 12, (3), [5], np.array([7]), (8, 3), [1, 12], np.array([5, 7, 1])]


y_simple = pd.DataFrame({"a": [1, 3, -1.5, -7]})
y_simple_expected_diff = {
    "drop_na": pd.DataFrame({"a": [2, -4.5, -5.5]}),
    "keep_na": pd.DataFrame({"a": [np.nan, -4.5, -5.5]}),
    "fill_zero": pd.DataFrame({"a": [0, -4.5, -5.5]}),
}


@pytest.mark.parametrize("na_handling", Differencer.VALID_NA_HANDLING_STR)
def test_differencer_produces_expected_results(na_handling):
    """Test that Differencer produces expected results on a simple DataFrame."""
    transformer = Differencer(na_handling=na_handling)
    y_transformed = transformer.fit_transform(y_simple)
    y_expected = y_simple_expected_diff[na_handling]

    _assert_array_almost_equal(y_transformed, y_expected)


@pytest.mark.parametrize("y", test_cases)
@pytest.mark.parametrize("lags", lags_to_test)
def test_differencer_same_series(y, lags):
    """Test transform against inverse_transform."""
    transformer = Differencer(lags=lags, na_handling="drop_na")
    y_transform = transformer.fit_transform(y)
    y_reconstructed = transformer.inverse_transform(y_transform)

    # Reconstruction should return the reconstructed series for same indices
    # that are in the `Z` timeseries passed to inverse_transform
    _assert_array_almost_equal(y.loc[y_reconstructed.index], y_reconstructed)


@pytest.mark.parametrize("na_handling", ["keep_na", "fill_zero"])
@pytest.mark.parametrize("y", test_cases)
@pytest.mark.parametrize("lags", lags_to_test)
def test_differencer_remove_missing_false(y, lags, na_handling):
    """Test transform against inverse_transform."""
    transformer = Differencer(lags=lags, na_handling=na_handling)
    y_transform = transformer.fit_transform(y)

    # if na_handling is fill_zero, get rid of the zeros for reconstruction
    if na_handling == "fill_zero":
        y_transform = y_transform[24:]
        y = y[24:]

    y_reconstructed = transformer.inverse_transform(y_transform)

    _assert_array_almost_equal(y, y_reconstructed)


@pytest.mark.parametrize("y", test_cases)
@pytest.mark.parametrize("lags", lags_to_test)
def test_differencer_prediction(y, lags):
    """Test transform against inverse_transform."""
    y_train = y.iloc[:-12].copy()
    y_true = y.iloc[-12:].copy()

    transformer = Differencer(lags=lags, na_handling="drop_na")
    y_transform = transformer.fit_transform(y)

    # Use the actual transformed values as predictions since we know we should
    # be able to convert them to the units of the original series and exactly
    # match the y_true values for this period
    y_pred = y_transform.iloc[-12:].copy()

    # Redo the transformer's fit and transformation
    # Now the transformer doesn't know anything about the values in y_true
    # This simulates use-case with a forecasting pipeline
    y_transform = transformer.fit_transform(y_train)

    y_pred_inv = transformer.inverse_transform(y_pred)

    _assert_array_almost_equal(y_true, y_pred_inv)


@pytest.mark.skipif(
    not _check_soft_dependencies("prophet", severity="none"),
    reason="requires Prophet forecaster in the example",
)
def test_differencer_cutoff():
    """Tests a special case that triggers freq inference.

    Failure mode:
    raises ValueError "Must supply freq for datetime value"
    on line "fh = ForecastingHorizon(etc" in Differencer._check_inverse_transform_index
    """
    from sktime.datasets import load_longley
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.fbprophet import Prophet
    from sktime.forecasting.model_selection import (
        ExpandingWindowSplitter,
        ForecastingGridSearchCV,
        temporal_train_test_split,
    )
    from sktime.transformations.series.difference import Differencer

    y, X = load_longley()

    # split train/test both y and X
    fh = [1, 2]
    train_model, _ = temporal_train_test_split(y, fh=fh)
    X_train = X[X.index.isin(train_model.index)]
    train_model.index = train_model.index.to_timestamp(freq="A")
    X_train.index = X_train.index.to_timestamp(freq="A")

    # pipeline
    pipe = TransformedTargetForecaster(
        steps=[
            ("differencer", Differencer(na_handling="fill_zero")),
            ("myforecaster", Prophet()),
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

    param_grid = [{"differencer__na_handling": ["fill_zero"]}]

    # grid search
    gscv = ForecastingGridSearchCV(
        forecaster=pipe,
        cv=cv,
        param_grid=param_grid,
        verbose=1,
    )

    # fit
    gscv.fit(train_model, X=X_train)


def test_differencer_inverse_does_not_memorize():
    """Tests that differencer inverse always computes inverse via cumsum.

    Test case by ilkersigirci in #3345 (simplified)

    Failure mode:
    previous versions "remembered" the fit data, which can lead to unexpected
    output in the case of pipelining forecasters with a Differencer, see # 3345
    """
    import numpy as np

    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.transformations.series.difference import Differencer

    y = load_airline()

    y_train, y_test = temporal_train_test_split(y=y, test_size=30)
    fh_out = np.arange(1, len(y_test) + 1)
    fh_ins = ForecastingHorizon(y_train.index, is_relative=False)[1:]

    pipe = Differencer() * NaiveForecaster()

    pipe.fit(y=y_train)
    pipe_ins = pipe.predict(fh=fh_ins)
    pipe.predict(fh=fh_out)

    naive_model = NaiveForecaster()
    naive_model.fit(y=y_train)
    model_ins = naive_model.predict(fh=fh_ins)
    naive_model.predict(fh=fh_out)

    # pipe output should not be similar to train input
    assert not np.allclose(y_train[1:].to_numpy(), pipe_ins.to_numpy())

    # pipe output should be similar to model output
    assert np.allclose(pipe_ins.to_numpy(), model_ins.to_numpy())
    # (first element can be different)

    # model output should not be similar to train input
    assert not np.allclose(y_train[1:].to_numpy(), model_ins.to_numpy())
