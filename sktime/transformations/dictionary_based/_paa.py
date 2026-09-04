"""Piecewise Aggregate Approximation Transformer (PAA)."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["MatthewMiddlehurst"]


class PAAlegacy(BaseTransformer):
    """Piecewise Aggregate Approximation Transformer (PAA).

    Piecewise Aggregate Approximation reduces the number of time points
    in a time series by replacing each interval with its mean value.

    Parameters
    ----------
    num_intervals : int, default=8
        Number of intervals in the transformed time series.
    """

    _tags = {
        "authors": ["MatthewMiddlehurst"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "capability:categorical_in_X": False,
    }

    def __init__(self, num_intervals=8):
        self.num_intervals = num_intervals
        super().__init__()

    def set_num_intervals(self, n):
        """Set self.num_intervals to n."""
        self.num_intervals = n

    def _transform(self, X, y=None):
        """Transform data using Piecewise Aggregate Approximation.

        Parameters
        ----------
        X : pd.DataFrame
            Time series with time points in rows and variables in columns.

        Returns
        -------
        pd.DataFrame
            Transformed time series with ``num_intervals`` rows and
            the same number of columns as ``X``.
        """
        num_timepoints = X.shape[0]

        self._check_parameters(num_timepoints)

        transformed = []

        for column in X.columns:
            values = X[column].to_numpy()

            paa_values = self._perform_paa_along_dim(values)

            transformed.append(paa_values)

        result = np.column_stack(transformed)

        return pd.DataFrame(
            result,
            columns=X.columns,
        )

    def _perform_paa_along_dim(self, series):
        """Perform PAA on one time series.

        Parameters
        ----------
        series : np.ndarray
            One-dimensional time series.

        Returns
        -------
        np.ndarray
            PAA representation containing ``num_intervals`` values.
        """
        series = np.asarray(series, dtype=float)

        n = len(series)
        frame_length = n / self.num_intervals

        frames = np.zeros(self.num_intervals, dtype=float)

        for i in range(self.num_intervals):
            start = i * frame_length
            end = (i + 1) * frame_length

            total = 0.0

            # Determine all data points that overlap this interval
            first = int(np.floor(start))
            last = int(np.ceil(end))

            for j in range(first, min(last, n)):
                # Amount of data point j that belongs to this interval
                overlap_start = max(start, j)
                overlap_end = min(end, j + 1)

                overlap = max(0.0, overlap_end - overlap_start)

                total += series[j] * overlap

            frames[i] = total / frame_length

        return frames

    def _check_parameters(self, num_atts):
        """Check parameters of PAA.

        Parameters
        ----------
        num_atts : int
            Number of time points.
        """
        if not isinstance(self.num_intervals, int):
            raise TypeError(
                "num_intervals must be an 'int'. Found '"
                + type(self.num_intervals).__name__
                + "' instead."
            )

        if self.num_intervals <= 0:
            raise ValueError("num_intervals must have the value of at least 1")

        if self.num_intervals > num_atts:
            raise ValueError(
                "num_intervals cannot be higher than the time series length."
            )
