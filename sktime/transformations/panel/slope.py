"""Slope transformer."""
import math
import statistics

import numpy as np
import pandas as pd

from sktime.datatypes import convert
from sktime.transformations.base import BaseTransformer

__all__ = ["SlopeTransformer"]
__author__ = ["mloning"]


class SlopeTransformer(BaseTransformer):
    """Slope-by-segment transformation.

    Class to perform the Slope transformation on a time series
    dataframe. It splits a time series into num_intervals segments.
    Then within each segment, it performs a total least
    squares regression to extract the gradient of the segment.

    Parameters
    ----------
    num_intervals : int, number of approx equal segments
                    to split the time series into.
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
    }

    def __init__(self, num_intervals=8):
        self.num_intervals = num_intervals
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        # Get information about the dataframe
        n_timepoints = len(X.iloc[0, 0])
        num_instances = X.shape[0]
        col_names = X.columns

        self._check_parameters(n_timepoints)

        Xt = pd.DataFrame()

        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = convert(
                pd.DataFrame(X[x]),
                from_type="nested_univ",
                to_type="numpyflat",
                as_scitype="Panel",
            )

            # Calculate gradients
            transformedData = []
            for y in range(num_instances):
                res = self._get_gradients_of_lines(arr[y])
                transformedData.append(res)

            # Convert to Numpy array
            transformedData = np.asarray(transformedData)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            Xt[x] = colToAdd

        return Xt

    def _get_gradients_of_lines(self, X):
        """Get gradients of lines.

        Function to get the gradients of the line of best fits
        given a time series.

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length]

        Returns
        -------
        gradients : a numpy array of shape = [num_intervals].
                    It contains the gradients of the line of best fit
                    for each interval in a time series.
        """
        # Firstly, split the time series into approx equal length intervals
        splitTimeSeries = self._split_time_series(X)
        gradients = []

        for x in range(len(splitTimeSeries)):
            gradients.append(self._get_gradient(splitTimeSeries[x]))

        return gradients

    def _get_gradient(self, Y):
        """Get gradient of lines.

        Function to get the gradient of the line of best fit given a
        section of a time series.

        Equation adopted from:
        real-statistics.com/regression/total-least-squares

        Parameters
        ----------
        Y : a numpy array of shape = [interval_size]

        Returns
        -------
        m : an int corresponding to the gradient of the best fit line.
        """
        # Create a list that contains 1,2,3,4,...,len(Y) for the x coordinates.
        X = [(i + 1) for i in range(len(Y))]

        # Calculate the mean of both lists
        meanX = statistics.mean(X)
        meanY = statistics.mean(Y)

        # Calculate the list (yi-mean(y))^2
        yminYbar = [(y - meanY) ** 2 for y in Y]
        # Calculate the list (xi-mean(x))^2
        xminXbar = [(x - meanX) ** 2 for x in X]

        # Sum them to produce w.
        w = sum(yminYbar) - sum(xminXbar)

        # Calculate the list (xi-mean(x))*(yi-mean(y))
        temp = []
        for x in range(len(X)):
            temp.append((X[x] - meanX) * (Y[x] - meanY))

        # Sum it and multiply by 2 to calculate r
        r = 2 * sum(temp)

        if r == 0:
            # remove nans
            m = 0
        else:
            # Gradient is defined as (w+sqrt(w^2+r^2))/r
            m = (w + math.sqrt(w**2 + r**2)) / r

        return m

    def _split_time_series(self, X):
        """Split a time series into approximately equal intervals.

        Adopted from = https://stackoverflow.com/questions/2130016/
                       splitting-a-list-into-n-parts-of-approximately
                       -equal-length

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length]

        Returns
        -------
        output : a numpy array of shape = [num_intervals,interval_size]
        """
        avg = len(X) / float(self.num_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning) : int(beginning + avg)])
            beginning += avg

        return output

    def _check_parameters(self, n_timepoints):
        """Check values of parameters for Slope transformer.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError(
                    "num_intervals must have the value \
                                  of at least 1"
                )
            if self.num_intervals > n_timepoints:
                raise ValueError(
                    "num_intervals cannot be higher than \
                                  subsequence_length"
                )
        else:
            raise TypeError(
                "num_intervals must be an 'int'. Found '"
                + type(self.num_intervals).__name__
                + "'instead."
            )
