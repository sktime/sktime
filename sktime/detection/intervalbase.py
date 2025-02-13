
"""Interval-based changepoint detection for time series."""

__all__ = ["IntervalBasedChangePointDetector"]
__author__ = ["Artinong"]

import numpy as np
import pandas as pd
from itertools import groupby
from collections import deque
from sktime.base import BaseEstimator
from sktime.forecasting.base import BaseForecaster


class IntervalBasedChangePointDetector(BaseEstimator):
    """Change point detector that uses forecasting intervals to detect sustained changes."""

    def __init__(self, forecaster, coverage=0.95, min_violations=3, window_size=10, z_threshold=1.0):
        """
        Initialize the change point detector with the specified parameters.
        
        Parameters:
            forecaster (BaseForecaster): Forecaster object that supports prediction intervals.
            coverage (float): Coverage for the prediction intervals (between 0 and 1).
            min_violations (int): Minimum number of consecutive violations to detect a change point.
            window_size (int): Size of the sliding window for checking consecutive violations.
            z_threshold (float): Z-score threshold for predicting change point probabilities.
        """
        super().__init__()

        # Validate forecaster
        if not isinstance(forecaster, BaseForecaster):
            raise ValueError("forecaster must be a BaseForecaster instance")

        if not forecaster.get_tag("capability:pred_int"):
            raise ValueError("forecaster must support prediction intervals")

        # Validate other parameters
        if not 0 < coverage < 1:
            raise ValueError("coverage must be between 0 and 1")
            
        if min_violations < 1:
            raise ValueError("min_violations must be at least 1")
            
        if window_size < min_violations:
            raise ValueError("window_size must be greater than or equal to min_violations")
            
        self.forecaster = forecaster
        self.coverage = coverage
        self.min_violations = min_violations
        self.window_size = window_size
        self.z_threshold = z_threshold

    def _fit(self, X, y=None):
        """Fit the change point detector on the time series data."""
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(X)
        return self

    def _predict(self, X):
        """
        Predict change point labels based on forecast intervals.
        
        Parameters:
            X (pd.Series): The time series data.
        
        Returns:
            pd.Series: Series with 1s indicating detected change points, 0s otherwise.
        """
        fh = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))
        pred_int = self.forecaster_.predict_interval(fh=fh, coverage=self.coverage)
        
        # Extract lower and upper bounds of the prediction intervals
        lower = pred_int.xs("lower", level=-1, axis=1)
        upper = pred_int.xs("upper", level=-1, axis=1)
        
        # Mark points outside the interval as violations
        violations = (X < lower) | (X > upper)
        
        # Initialize the change points array
        change_points = pd.Series(0, index=X.index)
        
        # Sliding window to track consecutive violations
        violation_window = deque(maxlen=self.window_size)
        
        for i, violation in enumerate(violations):
            violation_window.append(violation)
            
            if len(violation_window) == self.window_size:
                # Count consecutive violations and flag change points
                consecutive_violations = self._count_max_consecutive_true(violation_window)
                if consecutive_violations >= self.min_violations:
                    change_points.iloc[i] = 1
        
        return change_points

    def _predict_proba(self, X):
        """
        Predict probabilities for change points using z-scores from the forecast intervals.
        
        Parameters:
            X (pd.Series): The time series data.
        
        Returns:
            pd.DataFrame: A DataFrame with the probabilities of change points.
        """
        fh = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))
        pred_int = self.forecaster_.predict_interval(fh=fh, coverage=self.coverage)

        # Extract the lower and upper bounds of the prediction intervals
        lower = pred_int.xs("lower", level=-1, axis=1)
        upper = pred_int.xs("upper", level=-1, axis=1)

        # Calculate the z-scores (distance from the center of the interval)
        center = (upper + lower) / 2
        width = upper - lower
        z_scores = np.abs(X - center) / (width / 2)
        
        # Initialize probabilities array
        probs = pd.Series(0.0, index=X.index)
        
        # Sliding window to track consecutive z-score violations
        z_score_window = deque(maxlen=self.window_size)
        
        for i, z_score in enumerate(z_scores):
            z_score_window.append(z_score)
            
            if len(z_score_window) == self.window_size:
                # Calculate probability based on consecutive z-score violations
                consecutive_violations = self._count_consecutive_violations(z_score_window)
                prob = consecutive_violations / self.min_violations
                prob = min(1.0, prob)
                probs.iloc[i] = prob
        
        return pd.DataFrame({"proba": probs}, index=X.index)

    def _count_max_consecutive_true(self, series):
        """Helper method to count max consecutive True values in a series."""
        counts = [sum(1 for _ in group) for value, group in groupby(series) if value]
        return max(counts) if counts else 0

    def _count_consecutive_violations(self, z_scores):
        """Helper method to count consecutive violations based on z-scores."""
        violations = (np.array(z_scores) > self.z_threshold).tolist()
        return self._count_max_consecutive_true(violations)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.
        
        Parameters:
            parameter_set (str): The name of the parameter set to return.
        
        Returns:
            dict: Dictionary of parameter values.
        """
        from sktime.forecasting.naive import NaiveForecaster

        params = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "coverage": 0.9,  # Use 90% coverage for prediction intervals
            "min_violations": 3,  # Require 3 consecutive violations for a change point
        }
        return params
