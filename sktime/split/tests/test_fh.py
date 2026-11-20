# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ForecastingHorizonSplitter."""

import numpy as np
import pandas as pd
import pytest

from sktime.split import ForecastingHorizonSplitter
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ForecastingHorizonSplitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "fh, expected_train, expected_test",
    [
        # regression test for #10580: the last point of the series must be
        # reachable by train or test, not silently dropped
        ([1, 2, 3], [0, 1, 2, 3, 4, 5, 6], [7, 8, 9]),
        # fh including 0 must yield a monotonically increasing test set,
        # with the cutoff point shared between train and test
        ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6], [6, 7, 8, 9]),
        # min_step > 1 must not be dropped from the tail either
        ([2, 5], [0, 1, 2, 3, 4], [6, 9]),
        # negative (in-sample) fh values overlap train, by design
        ([-2, 5], [0, 1, 2, 3, 4], [2, 9]),
    ],
)
def test_forecasting_horizon_splitter_relative(fh, expected_train, expected_test):
    """Test that relative fh splits use the full series, per issue #10580."""
    y = pd.DataFrame({"column": range(10)})
    splitter = ForecastingHorizonSplitter(fh=fh)
    fold_generator = splitter.split(y)

    train_ix, test_ix = next(fold_generator)

    np.testing.assert_array_equal(train_ix, expected_train)
    np.testing.assert_array_equal(test_ix, expected_test)

    with pytest.raises(StopIteration):
        next(fold_generator)
