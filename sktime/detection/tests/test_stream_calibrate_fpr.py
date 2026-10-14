"""Tests for StreamCalibrateFPR, alarms above a quantile of recent scores."""

__author__ = ["yash-sangwan"]

import math

import numpy as np
import pandas as pd
import pytest

from sktime.detection.base import BaseDetector
from sktime.detection.compose import StreamCalibrateFPR
from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)


class _FixedScores(BaseDetector):
    """Test detector with scores fixed in advance, by position in the stream.

    ``transform_scores`` returns the scores of the points of the last ``fit`` or
    ``update``, taken from ``scores`` by their position from the start of the stream.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": True,
        "capability:pretrain": True,
        "capability:update": True,
        "fit_is_empty": False,
    }

    def __init__(self, scores):
        self.scores = scores
        super().__init__()

    def _pretrain(self, X, y=None):
        self.pretrained_ = True
        return self

    def _fit(self, X, y=None):
        self.start_ = 0
        self.n_seen_ = len(X)
        return self

    def _update(self, X, y=None):
        self.start_ = self.n_seen_
        self.n_seen_ += len(X)
        return self

    def _predict(self, X):
        return BaseDetector._empty_sparse()

    def _transform_scores(self, X):
        scores = np.asarray(self.scores, dtype="float64")
        return pd.DataFrame(
            {"scores": scores[self.start_ : self.start_ + len(X)]}, index=X.index
        )


def _replay(scores, n_fit, fpr, window_length, chunk_size=1):
    """Fit on the first n_fit points, then update_predict the rest in chunks.

    Returns the wrapper, and the alarm positions counted from the first live point.
    """
    X = pd.DataFrame({"x": np.zeros(len(scores))})
    detector = StreamCalibrateFPR(
        _FixedScores(list(scores)), fpr=fpr, window_length=window_length
    )
    detector.fit(X.iloc[:n_fit])

    alarms = []
    for start in range(n_fit, len(X), chunk_size):
        y_chunk = detector.update_predict(X.iloc[start : start + chunk_size])
        alarms += [start - n_fit + int(iloc) for iloc in y_chunk["ilocs"]]
    return detector, alarms


def _reference_alarms(scores, n_fit, fpr, window_length):
    """Alarms by the rule written out plainly: fire if above the quantile, then add."""
    buffer = [s for s in scores[:n_fit] if not np.isnan(s)][-window_length:]
    position = math.ceil((1 - fpr) * (window_length - 1) - 1e-9)

    alarms = []
    for i, score in enumerate(scores[n_fit:]):
        if np.isnan(score):
            continue
        if len(buffer) == window_length and score > sorted(buffer)[position]:
            alarms.append(i)
        buffer = (buffer + [score])[-window_length:]
    return alarms


def _random_scores(n, n_nan=0, seed=0):
    """Uniform random scores, with n_nan of them set to NaN."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(size=n)
    scores[rng.choice(n, size=n_nan, replace=False)] = np.nan
    return scores


def test_check_estimator():
    """Test StreamCalibrateFPR passes the sktime estimator checks."""
    from sktime.utils.estimator_checks import check_estimator

    check_estimator(StreamCalibrateFPR, raise_exceptions=True)


def test_fires_above_the_quantile_of_the_buffer():
    """Test a point fires if its score is above the quantile, then joins the buffer.

    fpr 0.5 over 2 scores takes the larger score. The buffer starts as [0.1, 0.2]:
    0.3 > 0.2 fires; 0.25 < 0.3 and 0.1 < 0.3 do not; 0.5 > 0.25 fires.
    """
    scores = [0.1, 0.2, 0.3, 0.25, 0.1, 0.5]
    detector, alarms = _replay(scores, n_fit=2, fpr=0.5, window_length=2)

    assert alarms == [0, 3]
    assert list(detector.scores_buffer_) == [0.1, 0.5]
    assert detector.threshold_ == 0.5


def test_matches_the_rule_on_random_scores():
    """Test the alarms equal those of the rule written out plainly."""
    scores = _random_scores(300, n_nan=30, seed=1)
    _, alarms = _replay(scores, n_fit=30, fpr=0.1, window_length=20)

    assert alarms == _reference_alarms(scores, n_fit=30, fpr=0.1, window_length=20)
    assert len(alarms) > 0


@pytest.mark.parametrize("chunk_size", [7, 270])
def test_same_alarms_for_any_chunk_size(chunk_size):
    """Test chunks give the same alarms as point by point updates."""
    scores = _random_scores(300, n_nan=30, seed=2)
    _, point_by_point = _replay(scores, n_fit=30, fpr=0.1, window_length=20)
    _, chunked = _replay(
        scores, n_fit=30, fpr=0.1, window_length=20, chunk_size=chunk_size
    )

    assert chunked == point_by_point


def test_nothing_fires_before_the_buffer_is_full():
    """Test nothing fires until the buffer holds window_length scores.

    Each score is larger than all before it, so every point fires once the buffer
    is full. fit fills 3 of 10 places, so the first 7 live points cannot fire.
    """
    scores = np.arange(30) / 30
    detector, alarms = _replay(scores, n_fit=3, fpr=0.1, window_length=10)

    assert alarms == list(range(7, 27))

    fitted = StreamCalibrateFPR(_FixedScores(list(scores)), fpr=0.1, window_length=10)
    fitted.fit(pd.DataFrame({"x": np.zeros(3)}))
    assert np.isnan(fitted.threshold_)


def test_buffer_keeps_the_last_scores():
    """Test the buffer holds the last window_length scores, oldest first."""
    scores = _random_scores(100, seed=3)
    detector, _ = _replay(scores, n_fit=10, fpr=0.2, window_length=5)

    assert list(detector.scores_buffer_) == list(scores[-5:])


@pytest.mark.parametrize("n_low, fires", [(5, False), (10, True)])
def test_old_scores_leave_the_buffer(n_low, fires):
    """Test a level shift: a middle score fires only once the high scores are gone.

    The buffer starts with 10 scores of 0.9. After 10 scores of 0.1 it holds only
    0.1, and 0.5 is above it. After 5 of them, 0.9 is still in the buffer.
    """
    scores = [0.9] * 10 + [0.1] * n_low + [0.5]
    _, alarms = _replay(scores, n_fit=10, fpr=0.1, window_length=10)

    assert alarms == ([n_low] if fires else [])


def test_alarm_share_is_close_to_fpr():
    """Test about a share fpr of the points fire, on stationary random scores."""
    scores = _random_scores(20_000, seed=4)
    _, alarms = _replay(
        scores, n_fit=1_000, fpr=0.01, window_length=1_000, chunk_size=19_000
    )

    share = len(alarms) / 19_000
    assert 0.005 <= share <= 0.015


def test_nan_never_fires_and_never_enters_the_buffer():
    """Test NaN scores never fire, are not buffered, and do not count for burn-in.

    fit gives 2 scores of 4. The live 0.4 is the fourth score, so it fills the
    buffer but cannot fire. Then 0.9 > 0.4 and 0.95 > 0.9 fire.
    """
    nan = np.nan
    scores = [nan, nan, 0.1, 0.2] + [nan, 0.3, nan, 0.4, 0.9, nan, 0.95]
    detector, alarms = _replay(scores, n_fit=4, fpr=0.25, window_length=4)

    assert alarms == [4, 6]
    assert list(detector.scores_buffer_) == [0.3, 0.4, 0.9, 0.95]


def test_equal_scores_do_not_fire():
    """Test a score equal to the quantile does not fire, as the comparison is >."""
    scores = [0.5] * 40
    _, alarms = _replay(scores, n_fit=10, fpr=0.2, window_length=5)

    assert alarms == []


def test_fit_fills_the_buffer_and_predict_does_not_change_it():
    """Test fit only fills the buffer, and predict leaves the buffer as it is."""
    scores = _random_scores(60, seed=5)
    X = pd.DataFrame({"x": np.zeros(60)})
    detector = StreamCalibrateFPR(_FixedScores(list(scores)), fpr=0.2, window_length=5)

    detector.fit(X.iloc[:20])
    assert list(detector.scores_buffer_) == list(scores[15:20])

    buffer, threshold = list(detector.scores_buffer_), detector.threshold_
    first = detector.predict(X.iloc[:20])
    second = detector.predict(X.iloc[:20])
    assert first.equals(second)
    assert list(detector.scores_buffer_) == buffer
    assert detector.threshold_ == threshold

    detector.update(X.iloc[20:30])
    buffer = list(detector.scores_buffer_)
    first = detector.predict(X.iloc[20:30])
    second = detector.predict(X.iloc[20:30])
    assert first.equals(second)
    assert list(detector.scores_buffer_) == buffer


def test_predict_on_other_points_uses_the_current_threshold():
    """Test points not seen in update are compared with the current quantile."""
    scores = _random_scores(60, seed=6)
    X = pd.DataFrame({"x": np.zeros(60)})
    detector = StreamCalibrateFPR(_FixedScores(list(scores)), fpr=0.2, window_length=5)
    detector.fit(X.iloc[:20])
    detector.update(X.iloc[20:30])

    # the test detector scores these 5 points as the 5 points from position 20
    other = pd.DataFrame({"x": np.zeros(5)}, index=range(100, 105))
    buffer = list(detector.scores_buffer_)
    alarms = detector.predict(other)["ilocs"].tolist()

    expected = np.flatnonzero(scores[20:25] > detector.threshold_).tolist()
    assert alarms == expected
    assert list(detector.scores_buffer_) == buffer


def test_predict_on_a_different_chunk_with_the_same_index():
    """Test a chunk with the index of the last update, but other values, is not it.

    Chunk A is decided point by point in update: 0.3 > 0.2 fires, 0.25 < 0.3 does
    not, so its alarms are [0], and the quantile ends at 0.3. Chunk B has the same
    index and other values, so it is compared with that quantile: the test detector
    scores it 0.3 and 0.25 by position, and neither is above 0.3.
    """
    scores = [0.1, 0.2, 0.3, 0.25]
    X = pd.DataFrame({"x": np.zeros(4)})
    detector = StreamCalibrateFPR(_FixedScores(scores), fpr=0.5, window_length=2)
    detector.fit(X.iloc[:2])

    chunk_a = X.iloc[2:4]
    chunk_b = pd.DataFrame({"x": [1.0, 1.0]}, index=chunk_a.index)
    stored = detector.update_predict(chunk_a)["ilocs"].tolist()
    alarms_b = detector.predict(chunk_b)["ilocs"].tolist()

    expected_b = np.flatnonzero(np.array(scores[2:4]) > detector.threshold_).tolist()
    assert stored == [0]
    assert alarms_b == expected_b == []
    assert detector.predict(chunk_a)["ilocs"].tolist() == stored


def test_pretrain_is_passed_on_and_each_fit_clones_it():
    """Test pretrain pretrains a clone, and each fit starts from a new clone of it."""
    scores = _random_scores(40, seed=7)
    X = pd.DataFrame({"x": np.zeros(40)})
    panel = pd.DataFrame(
        {"x": np.zeros(8)},
        index=pd.MultiIndex.from_product(
            [["a", "b"], range(4)], names=["instance", "time"]
        ),
    )
    inner = _FixedScores(list(scores))
    detector = StreamCalibrateFPR(inner, fpr=0.5, window_length=2)

    detector.pretrain(panel)
    pretrained = detector.pretrained_detector_
    assert detector.state == "pretrained"
    assert pretrained is not inner and pretrained.pretrained_
    assert inner.state == "new"

    detector.fit(X.iloc[:10])
    first = detector.detector_
    assert first is not pretrained
    assert first.pretrained_ and first.state == "fitted"
    assert pretrained.state == "pretrained"

    detector.update_predict(X.iloc[10:20])
    detector.fit(X.iloc[:10])
    assert detector.detector_ is not first
    assert list(detector.scores_buffer_) == list(scores[8:10])

    clone = detector.clone()
    assert clone.pretrained_detector_.pretrained_
    assert not clone.is_fitted


@pytest.mark.parametrize(
    "fpr, window_length",
    [
        (0, 5),
        (1, 5),
        (1.5, 5),
        (np.nan, 5),
        (True, 5),
        (0.2, 4),
        (0.2, 5.0),
        (0.2, True),
    ],
)
def test_invalid_fpr_or_window_length_raise(fpr, window_length):
    """Test fpr outside (0, 1), or window_length below ceil(1 / fpr), raise."""
    detector = StreamCalibrateFPR(
        _FixedScores([0.0] * 10), fpr=fpr, window_length=window_length
    )

    with pytest.raises(ValueError, match="fpr|window_length"):
        detector.fit(pd.DataFrame({"x": np.zeros(10)}))


def test_detector_that_cannot_be_wrapped_raises():
    """Test a non-detector, a non-anomaly detector, and one without scores raise."""
    from sktime.detection.dummy import ZeroAnomalies, ZeroChangePoints

    X = pd.DataFrame({"x": np.zeros(10)})

    with pytest.raises(TypeError, match="sktime detector"):
        StreamCalibrateFPR("not a detector", fpr=0.5, window_length=2).fit(X)
    with pytest.raises(ValueError, match="anomaly detector"):
        StreamCalibrateFPR(ZeroChangePoints(), fpr=0.5, window_length=2).fit(X)
    with pytest.raises(TypeError, match="transform_scores"):
        StreamCalibrateFPR(ZeroAnomalies(), fpr=0.5, window_length=2).fit(X)
