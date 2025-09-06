"""Tests for MLPTorchClassifier."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning.mlp_torch import MLPTorchClassifier
from sktime.utils.dependencies import _check_soft_dependencies


def create_synthetic_timeseries_data(
    n_samples=100, n_features=10, n_timesteps=50, n_classes=3
):
    """Create synthetic time series data for testing."""
    # Create random time series data
    X = np.random.randn(n_samples, n_features, n_timesteps)

    # Create random labels
    y = np.random.randint(0, n_classes, size=n_samples)

    return X, y


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="PyTorch not available",
)
class TestMLPTorchClassifier:
    """Test class for MLPTorchClassifier."""

    def test_mlp_torch_classifier_basic(self):
        """Test basic functionality of MLPTorchClassifier."""
        # Create synthetic data
        X, y = create_synthetic_timeseries_data(
            n_samples=50, n_features=5, n_timesteps=20, n_classes=2
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create classifier
        classifier = MLPTorchClassifier(
            num_epochs=5, batch_size=8, random_state=42  # Small number for testing
        )

        # Test that we can create the classifier
        assert classifier is not None

        # Test that we can fit the classifier
        classifier.fit(X_train, y_train)

        # Test that we can make predictions
        y_pred = classifier.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test that predictions are valid
        assert all(pred in [0, 1] for pred in y_pred)  # Binary classification

        # Test predict_proba
        y_proba = classifier.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)  # 2 classes
        assert np.allclose(y_proba.sum(axis=1), 1.0)  # Probabilities sum to 1

        # Test accuracy (should be reasonable for random data)
        accuracy = accuracy_score(y_test, y_pred)
        assert 0.0 <= accuracy <= 1.0

    def test_mlp_torch_classifier_multiclass(self):
        """Test MLPTorchClassifier with multiple classes."""
        # Create synthetic data with 3 classes
        X, y = create_synthetic_timeseries_data(
            n_samples=60, n_features=3, n_timesteps=15, n_classes=3
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create classifier
        classifier = MLPTorchClassifier(num_epochs=5, batch_size=8, random_state=42)

        # Fit and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

        # Check predictions
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1, 2] for pred in y_pred)

        # Check probabilities
        assert y_proba.shape == (len(X_test), 3)  # 3 classes
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_mlp_torch_classifier_parameters(self):
        """Test MLPTorchClassifier with different parameters."""
        X, y = create_synthetic_timeseries_data(
            n_samples=40, n_features=4, n_timesteps=25, n_classes=2
        )

        # Test with different optimizers
        for optimizer in ["Adam", "SGD", "AdamW"]:
            classifier = MLPTorchClassifier(
                num_epochs=3,
                batch_size=4,
                optimizer=optimizer,
                lr=0.01,
                random_state=42,
            )

            # Should be able to fit without errors
            classifier.fit(X, y)
            y_pred = classifier.predict(X)
            assert len(y_pred) == len(y)

    def test_mlp_torch_classifier_configurable(self):
        """Test MLPTorchClassifier with different configuration options."""
        X, y = create_synthetic_timeseries_data(
            n_samples=30, n_features=3, n_timesteps=10, n_classes=2
        )

        # Test with custom hidden dimensions
        classifier_custom = MLPTorchClassifier(
            num_epochs=3,
            batch_size=4,
            hidden_dims=[100, 50],  # Custom architecture
            activation="tanh",  # Different activation
            dropout=0.2,  # Add dropout
            use_bias=False,  # No bias
            random_state=42,
        )

        # Should be able to fit without errors
        classifier_custom.fit(X, y)
        y_pred = classifier_custom.predict(X)
        assert len(y_pred) == len(y)

    def test_mlp_torch_classifier_error_handling(self):
        """Test error handling in MLPTorchClassifier."""
        classifier = MLPTorchClassifier()

        # Test with invalid input shape (2D instead of 3D)
        X_2d = np.random.randn(10, 5)
        y = np.random.randint(0, 2, size=10)

        with pytest.raises(ValueError, match="Expected 3D input"):
            classifier.fit(X_2d, y)

    def test_mlp_torch_classifier_get_test_params(self):
        """Test get_test_params method."""
        params = MLPTorchClassifier.get_test_params()
        assert isinstance(params, list)
        assert len(params) > 0

        # Test that we can create an instance with test params
        for param_set in params:
            classifier = MLPTorchClassifier(**param_set)
            assert classifier is not None
