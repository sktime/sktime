import warnings

import pandas as pd
import numpy as np

from sktime.detection.base import BaseDetector
from sktime.split import SlidingWindowSplitter
from sktime.detection.model_evaluation import evaluate
from sktime.exceptions import FitFailedWarning


class DummyFitEmptyDetector(BaseDetector):
    """Detector with fit_is_empty tag that would otherwise raise on predict.

    We set fit_is_empty=True so evaluate should suppress FitFailedWarning
    when fit/update/predict fail.
    """

    def __init__(self):
        super().__init__()
        # mark that fit is empty
        self.set_tags(**{"fit_is_empty": True})

    def _fit(self, X, y=None):
        # fit is empty -> nothing to do
        return self

    def _predict(self, X):
        # raise an error to simulate prediction failure in downstream code
        raise RuntimeError("predict failed")


def test_evaluate_suppresses_fitfailedwarning_for_fit_is_empty():
    X = pd.Series(np.arange(10))
    y = pd.DataFrame({"ilocs": [0, 5]})
    splitter = SlidingWindowSplitter(fh=[1], window_length=5, step_length=5)

    det = DummyFitEmptyDetector()

    # capture warnings and assert no FitFailedWarning is present
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = evaluate(det, splitter, X, y=y, strategy="refit")

    assert not any(isinstance(rec.message, FitFailedWarning) for rec in w), (
        "FitFailedWarning was emitted for detector with fit_is_empty tag"
    )
