#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test for SeasonalDummiesOneHot transformer."""

__author__ = ["ericjb"]
__all__ = []

import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.dummies import SeasonalDummiesOneHot
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_for_class([SeasonalDummiesOneHot])
    or _check_soft_dependencies("pandas<2.0.0", severity="none"),
    # pandas 2.0.0 does not accept ME freq
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_seasonal_dummies():
    date_range = pd.date_range(start="2022-01-01", periods=4, freq="ME")
    y = pd.Series([1, 2, 3, 4], index=date_range)
    transformer = SeasonalDummiesOneHot()
    X = transformer.fit_transform(y=y, X=None)
    expected_columns = ["Jan", "Feb", "Mar", "Apr"]
    X_expected = pd.DataFrame(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        columns=expected_columns,
        index=date_range,
    )
    X_expected = X_expected.astype(int)
    X_expected = X_expected.iloc[:, 1:]  # drop the first dummy
    assert X.equals(X_expected), "Test failed: X does not match X_expected."
