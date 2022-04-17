# -*- coding: utf-8 -*-
"""Test Featureizer."""
__author__ = ["aiwalter"]

from pandas.testing import assert_series_equal

from sktime.datasets import load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.compose import Featureizer
from sktime.transformations.series.exponent import ExponentTransformer

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)


def test_featureized_values():
    """Test against plain transformation.

    Test to check that the featureized values are same as if transformation
    is done without Featureizer.
    """
    lags = len(y_test)
    featureizer = Featureizer(ExponentTransformer(), lags=lags)
    featureizer.fit(X_train, y_train)
    X_hat = featureizer.transform(X_test, y_test)

    exp_transformer = ExponentTransformer()
    exp_transformer.fit(y_train[:-lags])
    y_hat = exp_transformer.transform(y_train[-lags:])
    assert_series_equal(
        X_hat["TOTEMP_ExponentTransformer"], y_hat, check_index=False, check_names=False
    )
