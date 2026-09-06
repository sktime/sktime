"""Piecewise Aggregate Approximation Transformer (PAA)."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["MatthewMiddlehurst"]


class PAAlegacy(BaseTransformer):
    """Piecewise Aggregate Approximation Transformer (PAA).

    (PAA) Piecewise Aggregate Approximation Transformer, as described in
    Eamonn Keogh, Kaushik Chakrabarti, Michael Pazzani, and Sharad Mehrotra.
    Dimensionality reduction for fast similarity search in large time series
    databases.
    Knowledge and information Systems, 3(3), 263-286, 2001.
    For each series reduce the dimensionality to num_intervals, where each
    value is the mean of values in
    the interval.

    TO DO: pythonise it to make it more efficient. Maybe check vs this version
            http://vigne.sh/posts/piecewise-aggregate-approx/
    Could have: Tune the interval size in fit somehow?

    Parameters
    ----------
    num_intervals   : int, dimension of the transformed data (default 8)
    """

    _tags = {
        "authors": ["MatthewMiddlehurst"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "capability:categorical_in_X": False,
    }

    def __init__(self, num_intervals=8):
        self.num_intervals = num_intervals
        super().__init__()

    def set_num_intervals(self, n):
        """Set self.num_intervals to n."""
        self.num_intervals = n

    # todo: looks like this just loops over series instances
    # so should be refactored to work on Series directly
    def _transform(self, X, y=None):
        """Transform data using Piecewise Aggregate Approximation.

        Parameters
        ----------
        X : pd.DataFrame
            Time series with time points in rows and variables in columns.

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero,
              second in column one etc.
        pd.DataFrame
            Transformed time series with ``num_intervals`` rows and
            the same number of columns as ``X``.
        """
        # Get information about the dataframe
        num_timepoints = X.shape[0]

        # Check the parameters are appropriate
        self._check_parameters(num_timepoints)

        transformed = []
        # On each dimension, perform PAA
        for column in X.columns:
            values = X[column].to_numpy()

            paa_values = self._perform_paa_along_dim(values)

            transformed.append(paa_values)

        # Combine the dimensions together
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
                # if the last frame was lost due to double imprecision
                overlap = max(0.0, overlap_end - overlap_start)

                total += series[j] * overlap

            frames[i] = total / frame_length

        return frames

    def _check_parameters(self, num_atts):
        """Check parameters of PAA.

        Function for checking the values of parameters inserted into PAA.
        For example, the number of subsequences cannot be larger than the
        time series length.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.

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
