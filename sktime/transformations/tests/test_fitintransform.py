#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Imputer functionality."""

__author__ = ["aiwalter"]
__all__ = []

from pandas.testing import assert_frame_equal

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.compose import FitInTransform
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils._testing.forecasting import make_forecasting_problem

y, X = make_forecasting_problem(make_X=True)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)


def test_transform_fitintransform():
    """Test fit/transform against ExponentTransformer."""
    fitintransform = FitInTransform(ExponentTransformer())
    fitintransform.fit(X=X_train, y=y_train)
    y_hat = fitintransform.transform(X=X_test, y=y_test)

    y_hat_expected = ExponentTransformer().fit_transform(X_test, y_test)
    assert_frame_equal(y_hat, y_hat_expected)


def test_inverse_transform_fitintransform():
    """Test inverse_transform against ExponentTransformer."""
    fitintransform = FitInTransform(ExponentTransformer())
    fitintransform.fit(X=X_train, y=y_train)
    _ = fitintransform.transform(X=X_test, y=y_test)

    exponent = ExponentTransformer().fit(X_test, y_test)

    y_inv = fitintransform.inverse_transform(X_train, y_train)
    y_inv_expected = exponent.inverse_transform(X_train, y_train)
    assert_frame_equal(y_inv, y_inv_expected)
