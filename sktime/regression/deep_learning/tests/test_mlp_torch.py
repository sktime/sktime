"""Tests for MLPTorchRegressor."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sktime.regression.deep_learning.mlp_torch import MLPTorchRegressor
from sktime.utils.dependencies import _check_soft_dependencies


def create_synthetic_timeseries_data(n_samples=100, n_features=10, n_timesteps=50):
    """Create synthetic time series data for testing."""
    # Create random time series data
    X = np.random.randn(n_samples, n_features, n_timesteps)

    # Create random continuous target values
    y = np.random.randn(n_samples)

    return X, y


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="PyTorch not available",
)
class TestMLPTorchRegressor:
    """Test class for MLPTorchRegressor."""

    def test_mlp_torch_regressor_basic(self):
        """Test basic functionality of MLPTorchRegressor."""
        # Create synthetic data
        X, y = create_synthetic_timeseries_data(
            n_samples=50, n_features=5, n_timesteps=20
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create regressor
        regressor = MLPTorchRegressor(
            n_epochs=10,  # Small number for testing
            batch_size=8,
            verbose=True,
            random_state=42,
        )

        # Test that we can create the regressor
        assert regressor is not None

        # Test that we can fit the regressor
        regressor.fit(X_train, y_train)

        # Test that we can make predictions
        y_pred = regressor.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test that predictions are reasonable (not all the same)
        assert len(np.unique(y_pred)) > 1

        # Test MSE (should be reasonable for random data)
        mse = mean_squared_error(y_test, y_pred)
        assert mse >= 0.0

    def test_mlp_torch_regressor_configurable(self):
        """Test MLPTorchRegressor with different configuration options."""
        X, y = create_synthetic_timeseries_data(
            n_samples=30, n_features=3, n_timesteps=10
        )

        # Test with custom hidden dimensions
        regressor_custom = MLPTorchRegressor(
            n_epochs=5,
            batch_size=4,
            hidden_dims=[100, 50],  # Custom architecture
            activation="tanh",  # Different activation
            dropout=0.2,  # Add dropout
            use_bias=False,  # No bias
            loss="mean_absolute_error",  # Different loss
            optimizer="AdamW",  # Different optimizer
            lr=0.01,  # Different learning rate
            random_state=42,
        )

        # Should be able to fit without errors
        regressor_custom.fit(X, y)
        y_pred = regressor_custom.predict(X)
        assert len(y_pred) == len(y)

    def test_mlp_torch_regressor_error_handling(self):
        """Test error handling in MLPTorchRegressor."""
        regressor = MLPTorchRegressor()

        # Test with invalid input shape (2D instead of 3D)
        X_2d = np.random.randn(10, 5)
        y = np.random.randn(10)

        with pytest.raises(ValueError, match="Expected 3D input"):
            regressor.fit(X_2d, y)

    def test_mlp_torch_regressor_loss_functions(self):
        """Test different loss functions."""
        X, y = create_synthetic_timeseries_data(
            n_samples=20, n_features=2, n_timesteps=10
        )

        # Test different loss functions
        for loss in ["mean_squared_error", "mean_absolute_error", "huber"]:
            regressor = MLPTorchRegressor(
                n_epochs=3, batch_size=4, loss=loss, random_state=42
            )

            # Should be able to fit without errors
            regressor.fit(X, y)
            y_pred = regressor.predict(X)
            assert len(y_pred) == len(y)

    def test_mlp_torch_regressor_optimizers(self):
        """Test different optimizers."""
        X, y = create_synthetic_timeseries_data(
            n_samples=20, n_features=2, n_timesteps=10
        )

        # Test different optimizers
        for optimizer in ["Adam", "SGD", "AdamW"]:
            regressor = MLPTorchRegressor(
                n_epochs=3, batch_size=4, optimizer=optimizer, random_state=42
            )

            # Should be able to fit without errors
            regressor.fit(X, y)
            y_pred = regressor.predict(X)
            assert len(y_pred) == len(y)

    def test_mlp_torch_regressor_get_test_params(self):
        """Test get_test_params method."""
        params = MLPTorchRegressor.get_test_params()
        assert isinstance(params, list)
        assert len(params) > 0

        # Test that we can create an instance with test params
        for param_set in params:
            regressor = MLPTorchRegressor(**param_set)
            assert regressor is not None
