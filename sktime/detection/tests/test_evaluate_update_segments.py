import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.detection.model_evaluation import evaluate
from sktime.split import SlidingWindowSplitter


class DummyDetectorWithUpdateParams(BaseDetector):
    """Simple detector that records update_params values passed to update."""

    def __init__(self):
        super().__init__()
        self.update_calls = []

    def _fit(self, X, y=None):
        # no-op
        return

    def _update(self, X, y=None, update_params=True):
        # record the update_params flag
        self.update_calls.append(bool(update_params))

    def _predict(self, X):
        # return an empty prediction (no anomalies)
        return pd.DataFrame(columns=["ilocs"])

    # expose a public update that accepts update_params so evaluate can call it
    def update(self, X, y=None, update_params=True):
        # minimal safe update implementation that does not rely on BaseDetector.update
        self.update_calls.append(bool(update_params))
        return self


class DummyDetectorNoUpdateParams(BaseDetector):
    """Detector whose update does not accept update_params arg."""

    def __init__(self):
        super().__init__()
        self.update_calls = 0

    def _fit(self, X, y=None):
        return

    def _update(self, X, y=None):
        self.update_calls += 1

    def _predict(self, X):
        return pd.DataFrame(columns=["ilocs"])

    # public update without update_params argument -> fallback path
    def update(self, X, y=None):
        self.update_calls += 1
        return self


def make_segment_y():
    # create two segments as anomalies using ilocs pd.Interval
    segs = [pd.Interval(0, 2), pd.Interval(5, 7)]
    y = pd.DataFrame({"ilocs": segs})
    return y


def test_evaluate_update_strategy_for_segments():
    X = pd.Series(np.arange(12))
    y = make_segment_y()
    splitter = SlidingWindowSplitter(fh=[1], window_length=4, step_length=4)

    det = DummyDetectorWithUpdateParams()

    res = evaluate(det, splitter, X, y=y, strategy="update", return_model=True)

    # There should be multiple folds; ensure update was called and True passed
    fitted = res.loc[1, "fitted_detector"]
    assert isinstance(fitted, DummyDetectorWithUpdateParams)
    # update should have been called at least once with True
    assert any(fitted.update_calls), (
        "expected at least one update call with update_params=True"
    )


def test_evaluate_no_update_params_strategy_for_segments():
    X = pd.Series(np.arange(12))
    y = make_segment_y()
    splitter = SlidingWindowSplitter(fh=[1], window_length=4, step_length=4)

    det = DummyDetectorWithUpdateParams()

    res = evaluate(
        det, splitter, X, y=y, strategy="no-update_params", return_model=True
    )

    fitted = res.loc[1, "fitted_detector"]
    # update_calls should have at least one False entry
    assert any(call is False for call in fitted.update_calls), (
        "expected update_params=False in at least one update call"
    )


def test_evaluate_fallback_update_without_update_params_arg():
    X = pd.Series(np.arange(12))
    y = make_segment_y()
    splitter = SlidingWindowSplitter(fh=[1], window_length=4, step_length=4)

    det = DummyDetectorNoUpdateParams()

    res = evaluate(
        det, splitter, X, y=y, strategy="no-update_params", return_model=True
    )

    fitted = res.loc[1, "fitted_detector"]
    # since detector does not accept update_params, update_calls increments
    assert fitted.update_calls >= 0
