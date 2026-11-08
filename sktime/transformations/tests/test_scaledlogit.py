# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ScaledLogit transform unit tests."""

__author__ = ["ltsaprounis"]

import numpy as np
import pandas as pd
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
        (10, 70, np.log(TEST_SERIES - 10) - np.log(70 - TEST_SERIES)),
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
@pytest.mark.parametrize("X", [np.array([1e-300]), np.array([1e-300, 1e299])])
@pytest.mark.parametrize("lower, upper", [(1e-310, 1e300), (-1e300, -1e-310)])
def test_scaledlogit_transform_extreme_ratio(X, lower, upper):
    """Interior values have finite logits even when their distance ratio overflows."""
    if upper < 0:
        X = -X
    expected = np.array([-1381.5510557965274, -2.1972245773362196])[: X.size]
    if upper < 0:
        expected = -expected
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = ScaledLogitTransformer(lower, upper).fit_transform(X)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected)


def test_scaledlogit_extreme_ratio_multivariate_frame():
    """Finite logits preserve the frame layout and leave caller data untouched."""
    X = pd.DataFrame(
        {"small": [1e-300, 1e-200], "large": [1e299, 5e299]},
        index=pd.date_range("2026-01-01", periods=2),
    )
    original = X.copy(deep=True)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = ScaledLogitTransformer(1e-310, 1e300).fit_transform(X)
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(result["large"], [-np.log(9.0), 0.0], atol=1e-12)
    pd.testing.assert_index_equal(result.index, X.index)
    pd.testing.assert_index_equal(result.columns, X.columns)
    pd.testing.assert_frame_equal(X, original)
