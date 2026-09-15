#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test for SeasonalDummiesOneHot transformer."""

__author__ = ["ericjb"]
__all__ = []

import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.dummies import SeasonalDummiesOneHot


@pytest.mark.skipif(
    not run_test_for_class([SeasonalDummiesOneHot])
    or _check_soft_dependencies("pandas<2.1.0", severity="none"),
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


@pytest.mark.skipif(
    not run_test_for_class([SeasonalDummiesOneHot])
    or _check_soft_dependencies("pandas<2.1.0", severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_seasonal_dummies_hourly():
    """Test hourly frequency, see issue #8840.

    ``PeriodIndex.freqstr`` is ``"h"`` rather than ``"H"`` from pandas 2.2 on, so
    matching it against ``"H"`` raised ``ValueError: Unsupported frequency: h``.
    """
    date_range = pd.date_range(start="2023-01-01", periods=24, freq="h")
    X = pd.DataFrame({"values": range(24)}, index=date_range)

    Xt = SeasonalDummiesOneHot(freq="h").fit_transform(X)

    # one dummy per hour, less the dropped first one, plus the original column
    assert list(Xt.columns) == ["values"] + [f"H{i}" for i in range(1, 24)]


@pytest.mark.skipif(
    not run_test_for_class([SeasonalDummiesOneHot])
    or _check_soft_dependencies("pandas<2.1.0", severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "freq,sp,expected_prefix",
    [
        ("h", 8760, "H"),
        ("D", 365, "D"),
        ("W", 52, "W"),
        ("M", 12, None),  # month dummies are named Jan, Feb, ...
        ("Q", 4, "Q"),
    ],
)
def test_seasonal_dummies_all_documented_freqs(freq, sp, expected_prefix):
    """Test every frequency the docstring documents, via both freq and sp.

    Anchored frequencies report ``freqstr`` with a suffix (``"W-SUN"``,
    ``"Q-DEC"``), so weekly and quarterly were rejected the same way hourly was.
    Both the ``freq`` and the ``sp`` route reach the same dispatch.
    """
    periods = 24 if freq == "h" else 800
    index_freq = "h" if freq == "h" else "D"
    date_range = pd.date_range(start="2023-01-01", periods=periods, freq=index_freq)
    X = pd.DataFrame({"values": range(periods)}, index=date_range)

    Xt_freq = SeasonalDummiesOneHot(freq=freq).fit_transform(X)
    Xt_sp = SeasonalDummiesOneHot(sp=sp).fit_transform(X)

    # the two routes must agree
    pd.testing.assert_frame_equal(Xt_freq, Xt_sp)

    # and dummies must actually have been added
    assert Xt_freq.shape[1] > 1
    if expected_prefix is not None:
        dummy_cols = [c for c in Xt_freq.columns if c != "values"]
        assert all(c.startswith(expected_prefix) for c in dummy_cols)
