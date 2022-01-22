# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ScaledLogit transform unit tests."""
import numpy as np
import pytest
from pandas.testing import assert_series_equal

from sktime.datasets import load_airline
from sktime.transformations.series.scaledlogit import ScaledLogitTransformer

TEST_SERIES = np.array([30, 40, 60])


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


# Tests that all cases have a consistent inverse transform
@pytest.mark.parametrize(
    "lower, upper", [(0, 700), (None, 700), (0, None), (None, None)]
)
def test_scaledlogit_consistent_invesre_transform(lower, upper):
    """Tests that all cases have a consistent inverse transform."""
    y = load_airline()
    transformer = ScaledLogitTransformer(lower, upper)
    y_transformed = transformer.fit_transform(y)
    y_restored = transformer.inverse_transform(y_transformed)
    assert_series_equal(y, y_restored, check_names=False)


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        (0, 300, "X should not have values greater than upper_bound"),
        (300, 700, "X should not have values lower than lower_bound"),
    ],
)
def test_scaledlogit_bound_errors(lower, upper, message):
    """Tests all exceptions."""
    y = load_airline()
    with pytest.raises(ValueError) as excinfo:
        ScaledLogitTransformer(lower, upper).fit_transform(y)
        assert message in str(excinfo.value)
