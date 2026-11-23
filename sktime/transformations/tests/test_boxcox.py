"""Tests for BoxCoxTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
__all__ = []

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.boxcox import BoxCoxTransformer


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_boxcox_against_scipy():
    """Test boxcox transformer vs the scipy boxvox function."""
    from scipy.stats import boxcox

    y = load_airline()

    t = BoxCoxTransformer()
    actual = t.fit_transform(y)

    excepted, expected_lambda = boxcox(y.values)

    np.testing.assert_array_equal(actual, excepted)
    assert t.lambda_ == expected_lambda


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("bounds", [(0, 1), (-1, 0), (-1, 2)])
@pytest.mark.parametrize(
    "method, sp", [("mle", None), ("pearsonr", None), ("guerrero", 5)]
)
def test_lambda_bounds(bounds, method, sp):
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method=method, sp=sp)
    t.fit(y)
    assert bounds[0] < t.lambda_ < bounds[1]


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "bounds, r_lambda",
    [
        ((0, 1), 6.61069613518961e-05),
        ((-1, 0), -0.156975034727656),
        ((-1, 2), -0.156981228426408),
    ],
)
def test_guerrero_against_r_implementation(bounds, r_lambda):
    """Test BoxCoxTransformer against forecast guerrero method.

    Testing lambda values estimated by the R implementation of the Guerrero method
    https://github.com/robjhyndman/forecast/blob/master/R/guerrero.R
    against the guerrero method in BoxCoxTransformer.
    R code to generate the hardcoded value for bounds=(-1, 2) used in the test
    ('Airline.csv' contains the data from 'load_airline()'):
        airline_file <- read.csv(file = 'Airline.csv')[,c('Passengers')]
        airline.ts <- ts(airline_file)
        guerrero(airline.ts, lower=-1, upper=2, nonseasonal.length = 20)
    Output:
        -0.156981228426408
    """
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method="guerrero", sp=20)
    t.fit(y)
    np.testing.assert_almost_equal(t.lambda_, r_lambda, decimal=4)


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("lambda_fixed", [-1, 0, 0.5, 1, 2])
@pytest.mark.parametrize("enforce_positive", [False, True])
@pytest.mark.parametrize("as_series", [False, True])
def test_boxcox_enforce_positive(lambda_fixed, enforce_positive, as_series):
    """Only take absolute values when enforce_positive is enabled."""
    import pandas as pd
    from scipy.special import boxcox

    values = np.array([-4.0, -2.0, 0.5, 2.0, 4.0])
    X = pd.Series(values) if as_series else values
    transformer = BoxCoxTransformer(
        method="fixed",
        lambda_fixed=lambda_fixed,
        enforce_positive=enforce_positive,
    )

    actual = transformer.fit_transform(X)
    expected = boxcox(values, lambda_fixed)
    if enforce_positive:
        expected = np.sign(values) * boxcox(np.abs(values), lambda_fixed)

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.skipif(
    not run_test_for_class(BoxCoxTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["mle", "pearsonr", "guerrero"])
@pytest.mark.parametrize("enforce_positive", [False, True])
def test_boxcox_enforce_positive_after_fit(method, enforce_positive):
    """Respect enforce_positive when transforming after estimating lambda."""
    from scipy.special import boxcox

    X_train = np.array([1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 7.0, 11.0])
    X_test = np.array([-4.0, -2.0, 0.5, 2.0, 4.0])
    transformer = BoxCoxTransformer(
        method=method, sp=2, enforce_positive=enforce_positive
    ).fit(X_train)

    actual = transformer.transform(X_test)
    expected = boxcox(X_test, transformer.lambda_)
    if enforce_positive:
        expected = np.sign(X_test) * boxcox(np.abs(X_test), transformer.lambda_)

    np.testing.assert_allclose(actual, expected, equal_nan=True)
