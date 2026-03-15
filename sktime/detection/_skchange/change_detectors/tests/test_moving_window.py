"""Tests for MovingWindow and all available scores."""

import numpy as np
import pytest

from sktime.detection._skchange.anomaly_scores import LocalAnomalyScore
from sktime.detection._skchange.base import BaseIntervalScorer
from sktime.detection._skchange.change_detectors import MovingWindow
from sktime.detection._skchange.change_scores import CHANGE_SCORES, ContinuousLinearTrendScore, RankScore
from sktime.detection._skchange.costs import COSTS, RankCost
from sktime.detection._skchange.datasets import generate_alternating_data
from sktime.detection._skchange.tests.test_all_interval_scorers import skip_if_no_test_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


@pytest.mark.parametrize(
    "ScoreType, params",
    [
        (ScoreType, (bandwidth, selection_method))
        for ScoreType in SCORES_AND_COSTS
        for bandwidth, selection_method in [
            (12, "detection_length"),
            (12, "local_optimum"),
            [[12, 15, 20], "local_optimum"],
        ]
    ],
)
def test_moving_window_changepoint(ScoreType: type[BaseIntervalScorer], params: tuple):
    """Test MovingWindow changepoints with different parameters."""
    bandwidth, selection_method = params
    score = ScoreType.create_test_instance()
    skip_if_no_test_data(score)

    n_segments = 2
    seg_len = 100
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )

    # RankScore needs penalty 10 to detect single change in mean.
    if isinstance(score, RankScore) or isinstance(score, RankCost):
        penalty = 10.0
    else:
        penalty = 20.0

    detector = MovingWindow.create_test_instance().set_params(
        change_score=score,
        bandwidth=bandwidth,
        selection_method=selection_method,
        penalty=penalty,
    )

    changepoints = detector.fit_predict(df)["ilocs"]
    if isinstance(score, ContinuousLinearTrendScore):
        # ContinuousLinearTrendScore finds two changes in linear trend
        # (flat, steep, flat) instead of a single change in mean.
        assert len(changepoints) >= n_segments and len(changepoints) <= n_segments + 5
    else:
        assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len


def test_moving_window_continuous_linear_trend_score():
    """Test MovingWindow finds two change points with ContinuousLinearTrendScore."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    score = ContinuousLinearTrendScore.create_test_instance()
    detector = MovingWindow(score)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert len(changepoints) == n_segments


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_moving_window_scores(Score):
    """Test MovingWindow scores."""
    score = Score.create_test_instance()
    skip_if_no_test_data(score)
    if score.get_tag("is_penalised"):
        # Penalised scores cannot be forced to be non-negative in this test.
        pytest.skip(f"{type(score).__name__} is penalised.")

    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = MovingWindow(score, penalty=0)
    scores = detector.fit(df).transform_scores(df)
    assert np.all(np.logical_or(scores >= 0.0, np.isnan(scores)))
    assert len(scores) == len(df)


def test_invalid_change_scores():
    """
    Test that MovingWindow raises an error when given an invalid cost argument.
    """
    with pytest.raises(ValueError, match="change_score"):
        MovingWindow("l2")
    with pytest.raises(ValueError, match="change_score"):
        MovingWindow(LocalAnomalyScore(COSTS[1].create_test_instance()))


def test_invalid_bandwidth():
    """
    Test that MovingWindow raises an error when given an invalid bandwidth argument.
    """
    score = CHANGE_SCORES[0].create_test_instance()
    with pytest.raises(ValueError, match="bandwidth"):
        MovingWindow(score, bandwidth=0)
    with pytest.raises(ValueError, match="bandwidth"):
        MovingWindow(score, bandwidth=[])
    with pytest.raises(TypeError, match="bandwidth"):
        MovingWindow(score, bandwidth=1.5)
    with pytest.raises(TypeError, match="bandwidth"):
        MovingWindow(score, bandwidth=[1, 1.5])
    with pytest.raises(ValueError, match="bandwidth"):
        MovingWindow(score, bandwidth=[1, -1, 3])


def test_invalid_selection_method():
    """
    Test that MovingWindow raises an error when given an invalid selection_method.
    """
    score = CHANGE_SCORES[0].create_test_instance()
    with pytest.raises(ValueError, match="selection_method"):
        MovingWindow(score, selection_method="invalid_method")
    with pytest.raises(ValueError, match="multiple bandwidths"):
        MovingWindow(score, bandwidth=[2, 5], selection_method="detection_length")
