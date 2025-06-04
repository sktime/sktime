"""Smoke-test that tsai wrappers import, fit, and predict without error."""
import pandas as pd
import numpy as np
import pytest

from sktime.datasets import load_basic_motions
from sktime.tsai_integration.delegated_classifier import (
    TsaiTSTClassifier,
    TsaiInceptionTimeClassifier,
)

@pytest.mark.parametrize("Cls", [TsaiTSTClassifier, TsaiInceptionTimeClassifier])
def test_tsai_classifier_smoke(Cls):
    # load a tiny dataset (5 train instances, 1 channel, ~100 length)
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    clf = Cls(n_epochs=1, bs=4)   # fast smoke test
    clf.fit(X_train, y_train)

    preds = clf.predict(X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]
