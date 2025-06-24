# sktime/classification/deep_learning/tests/test_tsai_wrappers.py
import numpy as np
import pytest

from sktime.classification.deep_learning import (
    InceptionTimeClassifierTsai,
    TSTClassifierTsai,
)
from sktime.datasets import load_basic_motions


@pytest.mark.parametrize("Cls", [InceptionTimeClassifierTsai, TSTClassifierTsai])
def test_tsai_classifier_smoke(Cls):
    # load a tiny dataset (5 train instances, 1 channel, ~100 length)
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)

    # Instantiate with 1 epoch & small batch for speed
    clf = Cls(n_epochs=1, bs=4)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]
