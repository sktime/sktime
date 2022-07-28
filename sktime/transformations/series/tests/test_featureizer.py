# -*- coding: utf-8 -*-
"""Test Featurizer."""
__author__ = ["aiwalter"]

from numpy.testing import assert_array_equal

from sktime.datasets import load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.compose import Featurizer
from sktime.transformations.series.exponent import ExponentTransformer

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)


def test_featurized_values():
    """Test against plain transformation.

    Test to check that the featurized values are same as if transformation
    is done without Featurizer.
    """
    lags = len(y_test)
    featurizer = Featurizer(ExponentTransformer(), lags=lags)
    featurizer.fit(X_train, y_train)
    X_hat = featurizer.transform(X_test, y_test)

    exp_transformer = ExponentTransformer()
    exp_transformer.fit(y_train[:-lags])
    y_hat = exp_transformer.transform(y_train[-lags:])
    assert_array_equal(X_hat["TOTEMP_ExponentTransformer"].values, y_hat.values)
