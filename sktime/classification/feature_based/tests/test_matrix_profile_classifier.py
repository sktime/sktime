"""Regression tests for MatrixProfileClassifier."""

import numpy as np
import pytest

from sktime.classification.feature_based import MatrixProfileClassifier


@pytest.mark.parametrize("series_length", [1, 3])
def test_fit_rejects_subsequence_longer_than_series(series_length):
    """The last axis of a numpy3D panel is the length of each series."""
    X = np.arange(4 * series_length).reshape(4, 1, series_length)
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match=rf"m=4.*series length \({series_length}\)"):
        MatrixProfileClassifier(subsequence_length=4).fit(X, y)


def test_predict_rejects_subsequence_longer_than_series():
    """The prediction panel can be shorter than the training panel."""
    X = np.random.default_rng(42).normal(size=(4, 1, 12))
    y = np.array([0, 1, 0, 1])
    classifier = MatrixProfileClassifier(subsequence_length=4).fit(X, y)

    assert classifier.predict(X).shape == y.shape
    with pytest.raises(ValueError, match=r"m=4.*series length \(1\)"):
        classifier.predict(X[..., :1])
