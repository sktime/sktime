"""Tests for the PyTorch ResNet Classifier."""

__author__ = ["Vaseem34"]

import numpy as np
import pytest

from sktime.classification.deep_learning.resnet_torch import ResNetClassifier


@pytest.mark.parametrize("n_epochs", [1, 2])
def test_resnet_torch_integration(n_epochs):
    """Test the full lifecycle of the ResNetClassifier.

    This test verifies that the classifier can fit to data and produce
    predictions of the correct shape and type.
    """
    # 1. Create dummy data (3 samples, 1 channel, 50 time-steps)
    X = np.random.randn(3, 1, 50)
    y = np.array([0, 1, 0])

    # 2. Initialize the classifier with minimal epochs for speed
    clf = ResNetClassifier(n_epochs=n_epochs, batch_size=2)

    # 3. Test Fit
    clf.fit(X, y)

    # 4. Test Predict
    y_pred = clf.predict(X)

    # 5. Assertions (The technical check)
    assert isinstance(y_pred, np.ndarray), "Output should be a numpy array"
    assert len(y_pred) == 3, "Output length should match input samples"
    assert set(y_pred).issubset({0, 1}), "Predictions should be valid class labels"


if __name__ == "__main__":
    # This allows running the test manually via python test_resnet_torch.py
    test_resnet_torch_integration(n_epochs=1)
