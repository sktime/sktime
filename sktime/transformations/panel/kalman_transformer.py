"""Kalman Transformer for Panel Data using simdkalman.

This module implements a panel transformation using simdkalman.
It provides smoothing and filtering capabilities.
"""

import numpy as np
import pandas as pd
from simdkalman import KalmanFilter

from sktime.transformations.base import BaseTransformer


class SimdKalmanTransformer(BaseTransformer):
    """Kalman filter transformation for panel data using simdkalman.

    Parameters
    ----------
    transition_matrices : np.ndarray
        State transition matrix (F).
    observation_matrices : np.ndarray
        Observation matrix (H).
    transition_covariance : np.ndarray
        Process noise covariance (Q).
    observation_covariance : np.ndarray
        Measurement noise covariance (R).
    initial_state_mean : np.ndarray
        Initial state mean.
    initial_state_covariance : np.ndarray
        Initial state covariance matrix (P).
    """

    _tags = {
        "scitype:transform-output": "Panel",
        "requires-fh-in-fit": False,
        "capability:inverse_transform": True,
    }

    def __init__(
        self,
        transition_matrices,
        observation_matrices,
        transition_covariance,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
    ):
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance

        self.kf_ = None
        super().__init__()

    def _fit(self, X, y=None):
        """Initialize the Kalman filter model."""
        self.kf_ = KalmanFilter(
            state_transition=self.transition_matrices,
            process_noise=self.transition_covariance,
            observation_model=self.observation_matrices,
            observation_noise=self.observation_covariance,
        )
        return self

    def _transform(self, X, y=None):
        """Apply Kalman filter smoothing to the input panel data."""
        X_arr = X.to_numpy().reshape(1, *X.shape)  # Reshape to (N, T, D) format
        smoothed = self.kf_.smooth(X_arr)
        return pd.DataFrame(smoothed.means[0], index=X.index, columns=X.columns)

    def _inverse_transform(self, X, y=None):
        """Revert to original (not possible for Kalman, return X as is)."""
        return X

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameters."""
        return {
            "transition_matrices": np.array([[1]]),
            "observation_matrices": np.array([[1]]),
            "transition_covariance": np.array([[0.1]]),
            "observation_covariance": np.array([[1]]),
            "initial_state_mean": np.array([0]),
            "initial_state_covariance": np.array([[1]]),
        }
