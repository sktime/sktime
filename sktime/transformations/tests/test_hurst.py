# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Hurst exponent transformer."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.hurst import HurstExponentTransformer

pytestmark = pytest.mark.skipif(
    not run_test_for_class(HurstExponentTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _segmentwise_dfa(self, ts, lag):
    """Reference calculation before batching the segment fits."""
    y = ts.cumsum() - ts.mean()
    fluctuations = []
    for i in range(len(y) // lag):
        segment = y.iloc[i * lag : (i + 1) * lag]
        x = np.arange(lag)
        coef = np.polyfit(x, segment, 1)
        trend = np.polyval(coef, x)
        fluctuations.append(np.sqrt(np.mean((segment - trend) ** 2)))
    return np.mean(fluctuations)


@pytest.mark.parametrize("length", [32, 35])
@pytest.mark.parametrize("lag", [2, 3, 8, 16])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
@pytest.mark.parametrize("index_kind", ["range", "datetime"])
def test_dfa_matches_segmentwise_calculation(length, lag, dtype, index_kind):
    """Batching preserves complete segments, including with a discarded remainder."""
    values = np.random.default_rng(42).normal(size=length) * 10
    index = (
        pd.RangeIndex(100, 100 + 3 * length, 3)
        if index_kind == "range"
        else pd.date_range("2020-01-01", periods=length, freq="D")
    )
    ts = pd.Series(values.astype(dtype), index=index)
    transformer = HurstExponentTransformer(method="dfa")

    expected = _segmentwise_dfa(transformer, ts, lag)
    actual = transformer._calculate_dfa(ts, lag)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_dfa_linear_increments():
    """A three-point quadratic segment has a known detrended RMS fluctuation."""
    ts = pd.Series(np.arange(1, 13, dtype=float))
    actual = HurstExponentTransformer(method="dfa")._calculate_dfa(ts, lag=3)
    assert_allclose(actual, 1 / np.sqrt(18), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("length", [64, 67, 1024])
@pytest.mark.parametrize("lags", [None, [4, 8, 16]])
def test_dfa_fit_transform_matches_segmentwise_calculation(monkeypatch, length, lags):
    """The public result, including confidence bounds, matches the original fits."""
    ts = pd.Series(np.random.default_rng(42).normal(size=length).cumsum())
    actual = HurstExponentTransformer(method="dfa", lags=lags).fit_transform(ts)

    monkeypatch.setattr(HurstExponentTransformer, "_calculate_dfa", _segmentwise_dfa)
    expected = HurstExponentTransformer(method="dfa", lags=lags).fit_transform(ts)

    assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
