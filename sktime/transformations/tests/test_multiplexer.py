#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for MultiplexTransformer and associated dunders."""

__author__ = ["miraep8"]

import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.base import clone

from sktime.datasets import load_shampoo_sales
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.multiplexer import MultiplexTransformer
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils.validation.forecasting import check_scoring


def test_multiplex_transformer_alone():
    """Test behavior of MultiplexTransformer.

    Because MultiplexTransformer is in many ways a wrapper for an underlying
    transformer - we can confirm that if the selected_transformer is set that the
    MultiplexTransformer delegates all its transformation responsibilities as expected.
    """
    y = load_shampoo_sales()
    # randomly make some of the values nans:
    y.loc[y.sample(frac=0.1).index] = pd.np.nan
    # Note - we select two forecasters which are deterministic.
    transformer_tuples = [
        ("two", ExponentTransformer(2)),
        ("three", ExponentTransformer(3)),
    ]
    transformer_names = [name for name, _ in transformer_tuples]
    transformers = [transformer for _, transformer in transformer_tuples]
    multiplex_transformer = MultiplexTransformer(transformers=transformer_tuples)
    # for each of the forecasters - check that the wrapped forecaster predictions
    # agree with the unwrapped forecaster predictions!
    for ind, name in enumerate(transformer_names):
        # make a copy to ensure we don't reference the same objectL
        test_transformer = clone(transformers[ind])
        y_transform_indiv = test_transformer.fit_transform(X=y)
        multiplex_transformer.selected_transformer = name
        # Note- MultiplexForecaster will make a copy of the forecaster before fitting.
        y_transform_multi = multiplex_transformer.fit_transform(X=y)
        assert_array_equal(y_transform_indiv, y_transform_multi)


def _find_best_transformer(forecaster, transformers, cv, y):
    """Evaluate all the forecasters on y and return the name of best."""
    scoring = check_scoring(None)
    scoring_name = f"test_{scoring.name}"
    score = None
    for name, transformer in transformers:
        test_transformer = clone(transformer)
        y_hat = test_transformer.fit_transform(y)
        results = evaluate(clone(forecaster), cv, y_hat)
        results = results.mean()
        new_score = float(results[scoring_name])
        if not score or new_score < score:
            score = new_score
            best_name = name
    return best_name


def test_multiplex_transformer_in_grid():
    """Test behavior of MultiplexTransformer.

    It often makes sense to use MultiplexTransformer in conjunction with
    ForecastingGridSearchCV within a pipeline.  Here we check that when you do that
    you get the expected result.
    """
    y = load_shampoo_sales()
    # randomly make some of the values nans:
    y.iloc[[5, 10, 15, 25, 32]] = -1
    # Note - we select two forecasters which are deterministic.
    transformer_tuples = [
        ("two", ExponentTransformer(2)),
        ("three", ExponentTransformer(3)),
    ]
    transformer_names = [name for name, _ in transformer_tuples]
    multiplex_transformer = MultiplexTransformer(transformers=transformer_tuples)
    cv = ExpandingWindowSplitter(
        initial_window=24, step_length=12, start_with_window=True, fh=[1, 2, 3]
    )
    pipe = TransformedTargetForecaster(
        steps=[
            ("multiplex", multiplex_transformer),
            ("forecaster", NaiveForecaster(strategy="mean")),
        ]
    )
    gscv = ForecastingGridSearchCV(
        cv=cv,
        param_grid={"multiplex__selected_transformer": transformer_names},
        forecaster=pipe,
    )
    gscv.fit(y)
    best_steps = gscv.best_forecaster_.steps
    for name, estimator in best_steps:
        if "multiplex" == name:
            gscv_best_name = estimator.selected_transformer
    best_name = _find_best_transformer(
        NaiveForecaster(strategy="mean"), transformer_tuples, cv, y
    )
    assert gscv_best_name == best_name
