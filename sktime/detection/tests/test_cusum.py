"""Tests for the CUSUM change point detector."""

import pandas as pd
import pytest

from sktime.detection.cusum import CUSUM
from sktime.tests.test_switch import run_test_for_class

__author__ = ["RajdeepKushwaha5"]


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X, target, expected",
    [
        # single upward mean shift
        (pd.Series([0.0] * 20 + [5.0] * 20), 0.0, [19]),
        # single downward mean shift
        (pd.Series([5.0] * 20 + [0.0] * 20), 5.0, [19]),
        # two shifts: up then back down
        (pd.Series([0.0] * 20 + [5.0] * 15 + [0.0] * 20), 0.0, [19, 34]),
        # no change: constant series
        (pd.Series([0.0] * 30), 0.0, []),
    ],
)
def test_cusum_known_target(X, target, expected):
    """CUSUM detects exact change point ilocs for synthetic step signals."""
    model = CUSUM(k=0.5, h=4.0, target=target)
    result = model.fit_predict(X)
    assert result.values.flatten().tolist() == expected


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cusum_auto_target():
    """CUSUM estimates the baseline from warmup samples when target is None."""
    X = pd.Series([0.0] * 25 + [8.0] * 25)
    model = CUSUM(k=1.0, h=5.0, warmup_len=10)
    result = model.fit_predict(X)
    # warmup mean is 0.0; first sample of new level (iloc 25) raises C+ to 7 > 5
    assert result.values.flatten().tolist() == [24]


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cusum_high_threshold_no_detection():
    """CUSUM returns empty Series when threshold is never exceeded."""
    X = pd.Series([0.0] * 40)
    model = CUSUM(k=0.5, h=1000.0, target=0.0)
    result = model.fit_predict(X)
    assert len(result) == 0


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cusum_output_dtype():
    """fit_predict returns a DataFrame with an integer-typed 'ilocs' column."""
    X = pd.Series([0.0] * 20 + [5.0] * 20)
    model = CUSUM(k=0.5, h=4.0, target=0.0)
    result = model.fit_predict(X)
    assert isinstance(result, pd.DataFrame)
    assert result["ilocs"].dtype == "int64"


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cusum_single_element_series():
    """CUSUM on a length-1 series always returns empty (no left segment possible)."""
    X = pd.Series([99.0])
    model = CUSUM(k=0.5, h=0.001, target=0.0)
    result = model.fit_predict(X)
    assert len(result) == 0


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cusum_series_shorter_than_warmup():
    """CUSUM handles series shorter than warmup_len without error."""
    X = pd.Series([1.0, 2.0, 1.0])
    model = CUSUM(k=0.5, h=4.0, warmup_len=20)
    # should not raise; just returns empty or whatever the algorithm finds
    result = model.fit_predict(X)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cusum_ilocs_are_position_based():
    """Change point ilocs are positional even when the index is non-default."""
    # RangeIndex starting at 100 — the iloc of the change is still 19 (0-based position)
    X = pd.Series([0.0] * 20 + [5.0] * 20, index=range(100, 140))
    model = CUSUM(k=0.5, h=4.0, target=0.0)
    result = model.fit_predict(X)
    assert result.values.flatten().tolist() == [19]


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "bad_params, match",
    [
        ({"k": 0}, "k must be positive"),
        ({"k": -1.0}, "k must be positive"),
        ({"h": 0}, "h must be positive"),
        ({"h": -3.0}, "h must be positive"),
        ({"warmup_len": 0}, "warmup_len must be a positive integer"),
        ({"warmup_len": -5}, "warmup_len must be a positive integer"),
    ],
)
def test_cusum_invalid_params_raise(bad_params, match):
    """CUSUM raises ValueError for parameters that are out of valid range."""
    with pytest.raises(ValueError, match=match):
        CUSUM(**bad_params)
