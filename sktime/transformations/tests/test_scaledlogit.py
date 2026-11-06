# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ScaledLogit transform unit tests."""

__author__ = ["ltsaprounis"]

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.scaledlogit import ScaledLogitTransformer
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
    "lower, upper, values, expected",
    [
        (10, 70, [-1000.0, 0.0, 1000.0], [10.0, 40.0, 70.0]),
        (1e-20, 1e300, np.array([-710.0], dtype=np.float32), [4.476286225685e-9]),
        (
            np.nextafter(np.finfo(float).max, 0),
            np.finfo(float).max,
            [30.0],
            [np.finfo(float).max],
        ),
        (-np.finfo(float).max, np.finfo(float).max, [0.0], [0.0]),
        (1, 10**20, [0.0], [5e19]),
    ],
)
def test_scaledlogit_inverse_transform_extreme_values(lower, upper, values, expected):
    """The two-sided inverse stays finite and retains representable tail values."""
    transformer = ScaledLogitTransformer(lower_bound=lower, upper_bound=upper)
    transformer.fit(TEST_SERIES)

    with np.errstate(over="raise", invalid="raise"):
        result = transformer.inverse_transform(np.asarray(values))

    np.testing.assert_allclose(result, expected)


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
