#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Imputer functionality."""

__author__ = ["aiwalter"]
__all__ = []

from pandas.testing import assert_series_equal

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.compose import FitInTransform
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.utils._testing.forecasting import make_forecasting_problem

X = make_forecasting_problem()
X_train, X_test = temporal_train_test_split(X)


def test_transform_fitintransform():
    """Test fit/transform against BoxCoxTransformer."""
    fitintransform = FitInTransform(BoxCoxTransformer())
    fitintransform.fit(X=X_train)
    y_hat = fitintransform.transform(X=X_test)

    y_hat_expected = BoxCoxTransformer().fit_transform(X_test)
    assert_series_equal(y_hat, y_hat_expected)


def test_inverse_transform_fitintransform():
    """Test inverse_transform against BoxCoxTransformer."""
    fitintransform = FitInTransform(BoxCoxTransformer())
    fitintransform.fit(X=X_train)
    _ = fitintransform.transform(X=X_test)

    exponent = BoxCoxTransformer().fit(X_test)

    y_inv = fitintransform.inverse_transform(X_train)
    y_inv_expected = exponent.inverse_transform(X_train)
    assert_series_equal(y_inv, y_inv_expected)
