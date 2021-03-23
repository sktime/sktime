# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import statistics
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.validation.panel import check_X


class SlopeTransformer(_PanelToPanelTransformer):
    """
    Class to perform the Slope transformation on a time series
    dataframe. It splits a time series into num_intervals segments.
    Then within each segment, it performs a total least
    squares regression to extract the gradient of the segment.

    Parameters
    ----------
    num_intervals   :   int, number of approx equal segments
                        to split the time series into.
    """

    def __init__(self, num_intervals=8):
        self.num_intervals = num_intervals
        super(SlopeTransformer, self).__init__()

    def transform(self, X, y=None):

        """
        Parameters
        ----------
        X : a pandas dataframe of shape = [n_samples, num_dims]
            The training input samples.

        Returns
        -------
        df: a pandas data frame of shape = [num_intervals, num_dims]
        """

        # Check the data
        self.check_is_fitted()
        X = check_X(X, coerce_to_pandas=True)

        # Get information about the dataframe
        n_timepoints = len(X.iloc[0, 0])
        num_instances = X.shape[0]
        col_names = X.columns

        self._check_parameters(n_timepoints)

        df = pd.DataFrame()

        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = from_nested_to_2d_array(pd.DataFrame(X[x]), return_numpy=True)

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

            df[x] = colToAdd

        return df

    def _get_gradients_of_lines(self, X):

        """
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

        """
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
            m = (w + math.sqrt(w ** 2 + r ** 2)) / r

        return m

    def _split_time_series(self, X):
        """
        Function to split a time series into approximately equal intervals.

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
        """
        Function for checking the values of parameters inserted into Slope.

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
