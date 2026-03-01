"""Tests for the PELT change point detector."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.pelt import PELT
from sktime.tests.test_switch import run_test_for_class

__author__ = ["RajdeepKushwaha5"]

_SKIP = pytest.mark.skipif(
    not run_test_for_class(PELT),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@_SKIP
@pytest.mark.parametrize(
    "X, expected",
    [
        # single upward step
        (pd.Series([0.0] * 20 + [5.0] * 20, dtype=float), [19]),
        # single downward step
        (pd.Series([5.0] * 20 + [0.0] * 20, dtype=float), [19]),
        # two steps: up then back down
        (pd.Series([0.0] * 20 + [5.0] * 15 + [0.0] * 20, dtype=float), [19, 34]),
        # no change: constant series — no CPs reported
        (pd.Series([0.0] * 40, dtype=float), []),
    ],
)
def test_pelt_known_change_points(X, expected):
    """PELT returns exact change point ilocs for synthetic step signals."""
    model = PELT(penalty=2.0)
    result = model.fit_predict(X)
    assert result.values.flatten().tolist() == expected


@_SKIP
def test_pelt_flat_series_no_detection():
    """A perfectly flat series must yield no change points for any penalty."""
    X = pd.Series([3.14] * 50, dtype=float)
    result = PELT(penalty=1.0).fit_predict(X)
    assert len(result) == 0


@_SKIP
def test_pelt_high_penalty_fewer_cps():
    """Higher penalty should produce fewer (or equal) change points."""
    X = pd.Series([0.0] * 10 + [5.0] * 10 + [0.0] * 10 + [5.0] * 10, dtype=float)
    n_low = len(PELT(penalty=2.0).fit_predict(X))
    n_high = len(PELT(penalty=1e6).fit_predict(X))
    assert n_high <= n_low


@_SKIP
def test_pelt_single_cp_symmetric():
    """Symmetric up-then-down series: two CPs at the known positions."""
    X = pd.Series([0.0] * 15 + [5.0] * 15 + [0.0] * 15, dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    assert result.values.flatten().tolist() == [14, 29]


@_SKIP
def test_pelt_bic_penalty():
    """BIC penalty (2*log(n)) should still detect an obvious shift."""
    X = pd.Series([0.0] * 20 + [10.0] * 20, dtype=float)
    n = len(X)
    model = PELT(penalty=2 * np.log(n))
    result = model.fit_predict(X)
    assert 19 in result.values.flatten().tolist()


# ---------------------------------------------------------------------------
# Output-contract tests
# ---------------------------------------------------------------------------


@_SKIP
def test_pelt_output_type_and_column():
    """fit_predict returns a pd.DataFrame with an 'ilocs' column."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    assert isinstance(result, pd.DataFrame)
    assert "ilocs" in result.columns


@_SKIP
def test_pelt_output_sorted():
    """Change points are returned in ascending order."""
    X = pd.Series([0.0] * 10 + [4.0] * 10 + [-4.0] * 10 + [0.0] * 10, dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    ilocs = result.values.flatten().tolist()
    assert ilocs == sorted(ilocs)


@_SKIP
def test_pelt_output_dtype():
    """Change point values are integers (int64)."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    assert result["ilocs"].dtype == np.dtype("int64")


@_SKIP
def test_pelt_all_ilocs_in_bounds():
    """All returned ilocs must be valid indices of X."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    for iloc in result.values.flatten():
        assert 0 <= iloc < len(X)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


@_SKIP
def test_pelt_very_short_series():
    """Series shorter than 2 * min_cp_distance should not crash."""
    X = pd.Series([0.0, 1.0, 2.0], dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    assert isinstance(result, pd.DataFrame)


@_SKIP
def test_pelt_non_default_index_preserved_in_output_rows():
    """PELT output row count matches number of detected CPs regardless of index."""
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.Series([0.0] * 20 + [5.0] * 20, index=idx, dtype=float)
    result = PELT(penalty=2.0).fit_predict(X)
    assert len(result) == 1  # one change point expected


# ---------------------------------------------------------------------------
# Parameter-validation tests
# ---------------------------------------------------------------------------


@_SKIP
@pytest.mark.parametrize(
    "bad_params, match",
    [
        ({"penalty": -1.0}, "penalty must be non-negative"),
        ({"penalty": 2.0, "min_cp_distance": 0}, "min_cp_distance must be a positive"),
        ({"penalty": 2.0, "min_cp_distance": -3}, "min_cp_distance must be a positive"),
    ],
)
def test_pelt_invalid_params_raise(bad_params, match):
    """PELT raises ValueError for invalid constructor parameters."""
    with pytest.raises(ValueError, match=match):
        PELT(**bad_params)


# ---------------------------------------------------------------------------
# Consistency with BinarySegmentation
# ---------------------------------------------------------------------------


@_SKIP
def test_pelt_matches_bs_on_clean_signal():
    """On a clean two-segment signal PELT and BinarySegmentation agree."""
    from sktime.detection.bs import BinarySegmentation

    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    pelt_cp = PELT(penalty=2.0).fit_predict(X).values.flatten().tolist()
    bs_cp = BinarySegmentation(threshold=1).fit_predict(X).values.flatten().tolist()
    assert pelt_cp == bs_cp
