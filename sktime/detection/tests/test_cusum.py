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
        ({"warmup_len": 0}, "warmup_len must be a positive integer or None"),
        ({"warmup_len": -5}, "warmup_len must be a positive integer or None"),
        ({"warmup_len": 1.5}, "warmup_len must be a positive integer or None"),
    ],
)
def test_cusum_invalid_params_raise(bad_params, match):
    """CUSUM raises ValueError for parameters that are out of valid range."""
    with pytest.raises(ValueError, match=match):
        CUSUM(**bad_params)


# ---------------------------------------------------------------------------
# Score-method tests
# ---------------------------------------------------------------------------

# Numerical fixture used across several score tests.
# k=0.5, h=4.0, target=0  → alarm fires at t=21 (c_pos=5.0), CP iloc=20.
# running_scores: [0]*20, 2.5, 5.0 (pre-reset alarm value), 0.0, [0]*17
_X_SCORE = pd.Series([0.0] * 20 + [3.0] * 20, dtype=float)
_MODEL_SCORE = CUSUM(k=0.5, h=4.0, target=0.0)


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predict_scores_length_matches_change_points():
    """predict_scores returns one score per detected change point."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    cp_df = model.predict(_X_SCORE)
    scores_df = model.predict_scores(_X_SCORE)
    assert len(scores_df) == len(cp_df), (
        "predict_scores should have as many rows as predict"
    )


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predict_scores_values():
    """predict_scores alarm value equals max(C+,C-) at alarm step."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    scores_df = model.predict_scores(_X_SCORE)
    # One CP detected; alarm fired when c_pos first exceeded h=4.0 → 5.0
    assert len(scores_df) == 1
    assert abs(scores_df.iloc[0, 0] - 5.0) < 1e-9


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predict_scores_output_type():
    """predict_scores returns a pd.DataFrame (base class wraps to DataFrame)."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    scores_df = model.predict_scores(_X_SCORE)
    assert isinstance(scores_df, pd.DataFrame)


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predict_scores_empty_when_no_change_point():
    """predict_scores is empty when the series contains no change point."""
    model = CUSUM(k=0.5, h=4.0)
    flat = pd.Series([1.0] * 40, dtype=float)
    model.fit(flat)
    scores_df = model.predict_scores(flat)
    assert len(scores_df) == 0


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_length():
    """transform_scores returns one value per timepoint."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    ts = model.transform_scores(_X_SCORE)
    assert len(ts) == len(_X_SCORE)


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_index_preserved():
    """transform_scores preserves the original Series index."""
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.Series([0.0] * 20 + [3.0] * 20, index=idx, dtype=float)
    model = CUSUM(k=0.5, h=4.0, target=0.0)
    model.fit(X)
    ts = model.transform_scores(X)
    pd.testing.assert_index_equal(ts.index, idx)


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_pre_change_zeros():
    """Running statistic is 0 for all timepoints prior to the mean shift."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    ts = model.transform_scores(_X_SCORE)
    pre_change = ts.iloc[:20]
    assert (pre_change == 0.0).all(), "CUSUM statistic should be 0 before shift"


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_alarm_value_at_alarm_timepoint():
    """Running statistic at the alarm timepoint equals the pre-reset alarm score."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    ts = model.transform_scores(_X_SCORE)
    # t=21 is the alarm step; value must be 5.0 (above h=4.0), not 0
    assert abs(ts.iloc[21] - 5.0) < 1e-9


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_resets_after_alarm():
    """Running statistic resets to 0 on the timepoint *after* an alarm."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    ts = model.transform_scores(_X_SCORE)
    # t=21 is the alarm step; accumulator resets → statistic at t=22 must be 0
    assert ts.iloc[22] == 0.0


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_peak_before_alarm():
    """Running statistic at iloc 20 equals 2.5 (rising c_pos before alarm)."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    ts = model.transform_scores(_X_SCORE)
    assert abs(ts.iloc[20] - 2.5) < 1e-9


@pytest.mark.skipif(
    not run_test_for_class(CUSUM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_scores_output_type():
    """transform_scores returns a pd.Series."""
    model = _MODEL_SCORE.clone()
    model.fit(_X_SCORE)
    ts = model.transform_scores(_X_SCORE)
    assert isinstance(ts, pd.Series)
