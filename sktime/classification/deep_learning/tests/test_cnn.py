"""Tests for CNN classifier."""

import numpy as np
import pytest

from sktime.classification.deep_learning.cnn import CNNClassifier, CNNClassifierTorch
from sktime.datasets import load_unit_test


class TestCNNClassifier:
    """Test class for original CNNClassifier (TensorFlow version)."""

    def test_cnn_deprecation_message(self):
        """Test that original CNNClassifier shows helpful message."""
        with pytest.raises(NotImplementedError) as excinfo:
            CNNClassifier(n_epochs=1)

        # Check the message contains guidance to use PyTorch version
        assert "CNNClassifierTorch" in str(excinfo.value)


class TestCNNClassifierTorch:
    """Test class for CNNClassifierTorch (PyTorch implementation)."""

    def test_cnn_basic_functionality(self):
        """Test basic CNN functionality."""
        X_train, y_train = load_unit_test(split="train")
        X_test, y_test = load_unit_test(split="test")

        cnn = CNNClassifierTorch(n_epochs=2, batch_size=4, verbose=False)
        cnn.fit(X_train, y_train)

        y_pred = cnn.predict(X_test)
        y_proba = cnn.predict_proba(X_test)

        assert len(y_pred) == len(y_test)
        assert y_proba.shape[0] == len(y_test)
        assert y_proba.shape == len(cnn.classes_)

        # Check probability sums to 1
        prob_sums = np.sum(y_proba, axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6)

    @pytest.mark.parametrize("params", CNNClassifierTorch.get_test_params())
    def test_cnn_with_different_params(self, params):
        """Test CNN with different parameter configurations."""
        X_train, y_train = load_unit_test(split="train")

        cnn = CNNClassifierTorch(**params, verbose=False)
        cnn.fit(X_train, y_train)

        # Should not raise any errors
        assert hasattr(cnn, "model_")
        assert hasattr(cnn, "classes_")

    def test_cnn_pytorch_import(self):
        """Test that PyTorch imports work correctly."""
        # This should not raise an error if torch is installed
        cnn = CNNClassifierTorch(n_epochs=1, batch_size=2)
        assert cnn is not None
        assert cnn._tags["python_dependencies"] == "torch"
