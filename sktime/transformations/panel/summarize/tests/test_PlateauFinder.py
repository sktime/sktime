"""Tests for PlateauFinder."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.summarize import PlateauFinder


@pytest.mark.skipif(
    not run_test_for_class(PlateauFinder),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("value", [np.nan, -10, 10, -0.5, 0.5])
def test_PlateauFinder(value):
    """Test PlateauFinder on test data against expected output plateau features."""
    value = np.nan
    X = pd.DataFrame(
        pd.Series(
            [
                pd.Series([value, 1, 2, 3, value, 2, 2, 3]),  # nan at start
                pd.Series([value, value, 3, 3, value, 2, 2, 3]),
                pd.Series([0, value, value, value, value, value, 2, value]),
                # nan at end
                pd.Series([0, value, value, 3, 4, 5, value, value]),
                pd.Series([2, value, value, value, 2, value, 3, 1]),
                pd.Series([0, value, value, 3, value, value, 2, 0]),
            ]
        )
    )
    n_samples = X.shape[0]

    t = PlateauFinder(value=value, min_length=2)
    Xt = t.fit_transform(X)

    actual_starts = Xt.iloc[:, 0]
    actual_lengths = Xt.iloc[:, 1]

    expected_starts = [
        np.array([]),
        np.array([0]),
        np.array([1]),
        np.array([1, 6]),
        np.array([1]),
        np.array([1, 4]),
    ]
    expected_lengths = [
        np.array([]),
        np.array([2]),
        np.array([5]),
        np.array([2, 2]),
        np.array([3]),
        np.array([2, 2]),
    ]

    # compare results
    for i in range(n_samples):
        np.testing.assert_array_equal(actual_starts[i], expected_starts[i])
        np.testing.assert_array_equal(actual_lengths[i], expected_lengths[i])
