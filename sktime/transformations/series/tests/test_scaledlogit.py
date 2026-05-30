# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ScaledLogit transform unit tests."""

__author__ = ["ltsaprounis", "Akanksha Trehun"]

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.scaledlogit import ScaledLogitTransformer
from sktime.utils.warnings import warn

TEST_SERIES = np.array([30, 40, 60])


@pytest.mark.skipif(
    not run_test_for_class(ScaledLogitTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "lower, upper, output",
    [
        (10, 70, np.log((TEST_SERIES - 10) / (70 - TEST_SERIES))),
        (None, 70, -np.log(70 - TEST_SERIES)),
        (10, None, np.log(TEST_SERIES - 10)),
        (None, None, TEST_SERIES),
    ],
)
def test_scaledlogit_transform(lower, upper, output):
    """Test that we get the right output."""
    transformer = ScaledLogitTransformer(lower, upper)
    y_transformed = transformer.fit_transform(TEST_SERIES)
    assert np.all(output == y_transformed)


@pytest.mark.skipif(
    not run_test_for_class(ScaledLogitTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "lower, upper, message",
    [
        (
            0,
            300,
            (
                "X in ScaledLogitTransformer should not have values greater"
                "than upper_bound"
            ),
        ),
        (
            300,
            700,
            "X in ScaledLogitTransformer should not have values lower than lower_bound",
        ),
    ],
)
def test_scaledlogit_bound_errors(lower, upper, message):
    """Tests all exceptions."""
    y = load_airline()
    with pytest.warns(RuntimeWarning):
        ScaledLogitTransformer(lower, upper).fit_transform(y)
        warn(message, RuntimeWarning)


@pytest.mark.skipif(
    not run_test_for_class(ScaledLogitTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0, 70),   # lower_bound=0 is falsy — triggered the bug
        (0, 1),    # probability scale, lower=0
        (-10, 0),  # upper_bound=0 is falsy — triggered the bug
    ],
)
def test_scaledlogit_zero_bound_uses_scaled_logit(lower, upper):
    """Regression test: zero-valued bounds must use the scaled-logit branch.

    Failure case in bug #10003: `if self.upper_bound and self.lower_bound` was
    falsy when either bound was 0, causing the wrong transform branch to execute.
    """
    X = np.linspace(lower + 1e-3, upper - 1e-3, 10).reshape(1, -1)
    t = ScaledLogitTransformer(lower_bound=lower, upper_bound=upper)
    Xt = t.fit_transform(X)
    expected = np.log((X - lower) / (upper - X))
    np.testing.assert_allclose(Xt, expected, atol=1e-10)

    X_inv = t.inverse_transform(Xt)
    np.testing.assert_allclose(X_inv, X, atol=1e-10)
