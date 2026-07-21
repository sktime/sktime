"""Tests for the EWMA change point detector."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.ewma import EWMA
from sktime.tests.test_switch import run_test_for_class

__author__ = ["RajdeepKushwaha5"]

_SKIP = pytest.mark.skipif(
    not run_test_for_class(EWMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)

# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@_SKIP
@pytest.mark.parametrize(
    "X, target, expected",
    [
        # single upward shift: alarm fires 2 steps into new regime → CP=20
        (pd.Series([0.0] * 20 + [5.0] * 20, dtype=float), 0.0, [20]),
        # single downward shift: symmetric to upward → CP=20
        (pd.Series([5.0] * 20 + [0.0] * 20, dtype=float), 5.0, [20]),
        # two shifts: up then back down → CPs at 20 and 35
        # (EWMA alarm fires one step after z first exceeds h; reported CP is
        # the last  observation in the warmup of the new segment)
        (
            pd.Series([0.0] * 20 + [5.0] * 15 + [0.0] * 20, dtype=float),
            0.0,
            [20, 35],
        ),
        # no shift: flat series → no CPs
        (pd.Series([0.0] * 40, dtype=float), 0.0, []),
    ],
)
def test_ewma_known_target(X, target, expected):
    """EWMA returns correct change point ilocs for synthetic step signals."""
    model = EWMA(lam=0.2, h=1.0, target=target)
    result = model.fit_predict(X)
    assert result.values.flatten().tolist() == expected


@_SKIP
def test_ewma_auto_target():
    """EWMA estimates the baseline from warmup when target is None."""
    X = pd.Series([0.0] * 25 + [8.0] * 25)
    model = EWMA(lam=0.2, h=1.0, warmup_len=10)
    result = model.fit_predict(X)
    # warmup mean is 0.0; alarm fires quickly in the 8.0 regime
    ilocs = result.values.flatten().tolist()
    assert len(ilocs) == 1
    assert ilocs[0] >= 24  # alarm must be at or after the true shift point


@_SKIP
def test_ewma_high_threshold_no_detection():
    """EWMA returns an empty Series when the threshold is never exceeded."""
    X = pd.Series([0.0] * 40)
    model = EWMA(lam=0.2, h=1000.0, target=0.0)
    result = model.fit_predict(X)
    assert len(result) == 0


@_SKIP
def test_ewma_lam_one_is_threshold_detector():
    """lam=1 reduces EWMA to a raw point-level threshold on each observation."""
    # With lam=1, Z_t = x_t, so score = |x_t - mu|. Alarm fires when any
    # x_t deviates from mu by more than h.
    X = pd.Series([0.0] * 10 + [5.0] * 10, dtype=float)
    model = EWMA(lam=1.0, h=0.5, target=0.0)
    result = model.fit_predict(X)
    ilocs = result.values.flatten().tolist()
    # First observation > threshold is at index 10; alarm fires (t=10, t>0), CP=9
    assert ilocs == [9]


# ---------------------------------------------------------------------------
# Score-method tests
# ---------------------------------------------------------------------------


@_SKIP
def test_ewma_predict_scores_length_matches_cps():
    """predict_scores returns one score per detected change point."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0)
    model.fit(X)
    n_cp = len(model.predict(X))
    n_sc = len(model.predict_scores(X))
    assert n_cp == n_sc


@_SKIP
def test_ewma_predict_scores_value():
    """predict_scores returns |Z_t - mu| at the alarm timepoint."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0)
    model.fit(X)
    scores_df = model.predict_scores(X)
    assert len(scores_df) == 1
    # alarm fires at t=21 → z = lam*5 + (1-lam)*lam*5 = 1.8 → score = 1.8
    assert abs(scores_df.iloc[0, 0] - 1.8) < 1e-9


@_SKIP
def test_ewma_predict_scores_type():
    """predict_scores returns a pd.DataFrame (wraps _predict_scores result)."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0).fit(X)
    scores = model.predict_scores(X)
    assert isinstance(scores, pd.DataFrame)


@_SKIP
def test_ewma_predict_scores_empty_when_no_cp():
    """predict_scores is empty when no change point is detected."""
    X = pd.Series([2.0] * 40, dtype=float)
    model = EWMA(lam=0.2, h=1000.0).fit(X)
    assert len(model.predict_scores(X)) == 0


@_SKIP
def test_ewma_transform_scores_length():
    """transform_scores returns one value per observation."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0).fit(X)
    ts = model.transform_scores(X)
    assert len(ts) == len(X)


@_SKIP
def test_ewma_transform_scores_index_preserved():
    """transform_scores preserves the original Series index."""
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.Series([0.0] * 20 + [5.0] * 20, index=idx, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0).fit(X)
    ts = model.transform_scores(X)
    pd.testing.assert_index_equal(ts.index, idx)


@_SKIP
def test_ewma_transform_scores_pre_change_zero():
    """Running statistic is 0 for all timepoints in the flat pre-shift region."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0).fit(X)
    ts = model.transform_scores(X)
    assert (ts.iloc[:20] == 0.0).all()


@_SKIP
def test_ewma_transform_scores_resets_after_alarm():
    """Running statistic resets to 0.0 immediately after an alarm fires."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0).fit(X)
    ts = model.transform_scores(X)
    # alarm fires at t=21; running_scores[21] = 0 (reset)
    assert ts.iloc[21] == 0.0


@_SKIP
def test_ewma_transform_scores_type():
    """transform_scores returns a pd.Series."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    model = EWMA(lam=0.2, h=1.0, target=0.0).fit(X)
    ts = model.transform_scores(X)
    assert isinstance(ts, pd.Series)


# ---------------------------------------------------------------------------
# Output-contract tests
# ---------------------------------------------------------------------------


@_SKIP
def test_ewma_output_dtype():
    """Change point values are int64."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    result = EWMA(lam=0.2, h=1.0, target=0.0).fit_predict(X)
    assert result["ilocs"].dtype == np.dtype("int64")


@_SKIP
def test_ewma_output_type_and_column():
    """fit_predict returns a pd.DataFrame with an 'ilocs' column."""
    X = pd.Series([0.0] * 20 + [5.0] * 20, dtype=float)
    result = EWMA(lam=0.2, h=1.0, target=0.0).fit_predict(X)
    assert isinstance(result, pd.DataFrame)
    assert "ilocs" in result.columns


@_SKIP
def test_ewma_single_element_series():
    """EWMA does not crash on a length-1 Series."""
    X = pd.Series([3.14])
    result = EWMA(lam=0.2, h=1.0).fit_predict(X)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Parameter-validation tests
# ---------------------------------------------------------------------------


@_SKIP
@pytest.mark.parametrize(
    "bad_params, match",
    [
        ({"lam": 0.0}, "0 < lam"),
        ({"lam": -0.5}, "0 < lam"),
        ({"lam": 1.5}, "0 < lam"),
        ({"lam": 0.2, "h": 0}, "h must be positive"),
        ({"lam": 0.2, "h": -1.0}, "h must be positive"),
        ({"lam": 0.2, "warmup_len": 0}, "warmup_len must be a positive integer"),
        ({"lam": 0.2, "warmup_len": -3}, "warmup_len must be a positive integer"),
    ],
)
def test_ewma_invalid_params_raise(bad_params, match):
    """EWMA raises ValueError for out-of-range constructor parameters."""
    with pytest.raises(ValueError, match=match):
        EWMA(**bad_params)
