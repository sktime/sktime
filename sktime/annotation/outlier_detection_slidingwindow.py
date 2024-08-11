#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformers for detecting outliers in a time series."""

__author__ = ["yelenayu"]
__all__ = ["SlidingWindow"]


import warnings
import numpy as np
import pandas as pd
from sktime.split import SlidingWindowSplitter
from sktime.transformations.base import BaseTransformer


class MovingAverageZscoreOutlier(BaseTransformer):
    """Detect outliers using the moving average and modified z-score method with a sliding window.

    Parameters
    ----------
    window_length : int, optional (default=10)
        Length of the sliding window for moving average calculation
    threshold : float, optional (default=3.5)
        Threshold for the modified z-score to define an outlier
    return_bool : bool, optional (default=False)
        If True, outliers are marked with True and non-outliers with False.
        Else, outliers are marked with np.nan.

    Examples
    --------
    >>> from sktime.transformations.series.outlier_detection import MovingAverageModifiedZOutlier
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MovingAverageModifiedZOutlier(window_length=10)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "authors": ["yelenayu"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "handles-missing-data": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
    }

    def __init__(self, window_length=10, threshold=3.5, use_modified_z=True, return_bool=False):
        self.window_length = window_length
        self.threshold = threshold
        self.use_modified_z = use_modified_z
        self.return_bool = return_bool
        super().__init__()


    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Z : pd.Series or pd.DataFrame, same type as X
            Transformed version of X
        """
        Z = X.copy()

        if isinstance(Z, pd.DataFrame):
            for col in Z:
                Z[col] = self._transform_series(Z[col])
        else:
            Z = self._transform_series(Z)

        return Z

    def _transform_series(self, Z):
        """Transform the series Z by detecting outliers using moving average and z-score or modified z-score.

        Parameters
        ----------
        Z : pd.Series

        Returns
        -------
        pd.Series
        """
        if Z.isnull().values.any():
            warnings.warn(
                "Series contains nan values, more nan might be added if there are outliers",
                stacklevel=2,
            )

        Z_outliers = Z.copy()
        cv = SlidingWindowSplitter(
            fh=0,
            window_length=self.window_length,
            step_length=1,
            start_with_window=True,
        )

        for i in cv.split(Z):
            cv_window = i[0]
            window_data = Z.iloc[cv_window]

            if cv_window[0] < self.window_length:
                # Handle the edge case: first window- Use the data in this current window
                if self.use_modified_z:
                    window_median = window_data.median()
                    window_mad = np.median(np.abs(window_data - window_median))
                    z_scores = 0.6745 * (window_data - window_median) / window_mad
                else:
                    window_mean = window_data.mean()
                    window_std = window_data.std()
                    z_scores = (window_data - window_mean) / window_std

            else:
                # For subsequent windows, use the statistics from the previous window
                prev_window_data = Z.iloc[cv_window - 1]
                if self.use_modified_z:
                    prev_median = prev_window_data.median()
                    prev_mad = np.median(np.abs(prev_window_data - prev_median))
                    z_scores = 0.6745 * (window_data - prev_median) / prev_mad
                else:
                    prev_mean = prev_window_data.mean()
                    prev_std = prev_window_data.std()
                    z_scores = (window_data - prev_mean) / prev_std

            for idx in cv_window:
                if abs(z_scores[idx]) > self.threshold:
                    Z_outliers.iloc[idx] = np.nan
                elif self.return_bool:
                    Z_outliers.iloc[idx] = abs(z_scores[idx]) > self.threshold

        return Z_outliers


    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        param1 = {"window_length": 3}
        param2 = {"window_length": 5, "threshold": 2.5, "return_bool": True}
        return [param1, param2]