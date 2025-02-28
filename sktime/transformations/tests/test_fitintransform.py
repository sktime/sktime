#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of FitInTransform functionality."""

__author__ = ["aiwalter"]
__all__ = []

import pytest
from pandas.testing import assert_series_equal

from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import FitInTransform
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.skipif(
    not run_test_for_class([FitInTransform, BoxCoxTransformer]),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_transform_fitintransform():
    """Test fit/transform against BoxCoxTransformer."""
    X = make_forecasting_problem()
    X_train, X_test = temporal_train_test_split(X)

    fitintransform = FitInTransform(BoxCoxTransformer())
    fitintransform.fit(X=X_train)
    y_hat = fitintransform.transform(X=X_test)

    y_hat_expected = BoxCoxTransformer().fit_transform(X_test)
    assert_series_equal(y_hat, y_hat_expected)
