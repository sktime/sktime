#!/usr/bin/env python3 -u
"""Tests for hierarchical reconciler forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["RobKuebler"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.dummy import ForecastKnownValues
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.tests.test_switch import run_test_for_class


@pytest.fixture
def y_known():
    index = pd.MultiIndex.from_product(
        [["A", "B"], ["X"], [0, 1, 2]],
        names=["Level1", "Level2", "Date"],
    )
    data = range(len(index))

    return pd.DataFrame(data, index=index, columns=["Value"])


@pytest.mark.skipif(
    not run_test_for_class(ForecastKnownValues),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_multiindex(fh, y_known) -> None:
    """Test multiindex y_known."""
    f = ForecastKnownValues(y_known=y_known)
    f.fit(y_known)
    y_pred = f.predict(fh)

    # Create expected data
    if not isinstance(fh, np.ndarray):
        fh = np.array([fh])

    index = pd.MultiIndex.from_product(
        [["A", "B"], ["X"], fh + 2],
        names=["Level1", "Level2", "Date"],
    )
    expected = pd.DataFrame(None, index=index, columns=["Value"])

    pd.testing.assert_frame_equal(y_pred, expected, check_dtype=False)


@pytest.mark.skipif(
    not run_test_for_class(ForecastKnownValues),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_singleindex(fh, y_known) -> None:
    """Test singleindex y_known."""
    f = ForecastKnownValues(y_known=y_known.loc["A", "X"])
    f.fit(y_known.loc["A", "X"])
    y_pred = f.predict(fh)

    # Create expected data
    if not isinstance(fh, np.ndarray):
        fh = np.array([fh])

    expected = pd.DataFrame(None, index=fh + 2, columns=["Value"])

    pd.testing.assert_frame_equal(y_pred, expected, check_dtype=False)


@pytest.mark.skipif(
    not run_test_for_class(ForecastKnownValues),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("fill_value", [None, 1.0])
def test_fail_if_y_known_is_multiindex_but_y_is_not(fh, fill_value, y_known) -> None:
    """Test singleindex y_known."""
    f = ForecastKnownValues(y_known=y_known, fill_value=fill_value)
    f.fit(y_known.loc["A", "X"])
    y_pred = f.predict(fh)

    # Create expected data
    if not isinstance(fh, np.ndarray):
        fh = np.array([fh])

    expected = pd.DataFrame(fill_value, index=fh + 2, columns=["Value"]).astype(float)

    pd.testing.assert_frame_equal(y_pred, expected, check_dtype=False)
