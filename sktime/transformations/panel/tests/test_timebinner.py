"""Tests for TimeBinner."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.reduce import TimeBinner


@pytest.mark.skipif(
    not run_test_for_class(TimeBinner),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timebinner():
    """Test TimeBinner."""
    X, y = load_basic_motions(return_X_y=True)

    aggfunc = np.sum
    freq = 8
    idx = pd.interval_range(start=0, end=100, freq=freq, closed="left")
    tb = TimeBinner(idx=idx, aggfunc=aggfunc)
    tb.fit(X)
    row = 1
    Xtb = tb.transform(X)
    assert np.isclose(np.sum(X.iloc[row, 0][freq : (2 * freq)]), Xtb.iloc[row, 1])


@pytest.mark.skipif(
    not run_test_for_class(TimeBinner),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timebinner2():
    """Test TimeBinner."""
    X, y = load_basic_motions(return_X_y=True)

    aggfunc = np.max
    freq = 10
    idx = pd.interval_range(start=0, end=100, freq=freq, closed="right")
    tb = TimeBinner(idx=idx, aggfunc=aggfunc)
    tb.fit(X)
    row = 3
    Xtb = tb.transform(X)
    assert np.isclose(
        np.max(X.iloc[row, 5][8 * 10 + 1 : 9 * 10 + 1]), Xtb.iloc[row, 58]
    )


@pytest.mark.skipif(
    not run_test_for_class(TimeBinner),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timebinner3():
    """Test TimeBinner."""
    X, y = load_basic_motions(return_X_y=True)

    def aggfunc(series):
        return np.quantile(series, q=0.25)

    freq = 5
    idx = pd.interval_range(start=0, end=100, freq=freq, closed="right")
    tb = TimeBinner(idx=idx, aggfunc=aggfunc)
    row = 1
    col = 3
    tb.fit(X)
    Xtb = tb.transform(X)
    assert np.isclose(
        np.quantile(X.iloc[row, 0][col * freq + 1 : ((col + 1) * freq) + 1], q=0.25),
        Xtb.iloc[row, col],
    )
