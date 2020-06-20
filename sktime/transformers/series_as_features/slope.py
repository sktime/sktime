import numpy as np
import pandas as pd
import math
import statistics
from sktime.transformers.series_as_features.base \
    import BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X


class Slope(BaseSeriesAsFeaturesTransformer):

    def __init__(self, num_intervals=8):
        self.num_intervals = num_intervals
        super(Slope, self).__init__()

    """
    Parameters
    ----------
    X : a pandas dataframe of shape = [n_samples, num_dims]
        The training input samples.

    Returns
    -------
    dims: a pandas data frame of shape = [n_samples, num_dims]
    """
    def transform(self, X, y=None):
        # Check the data
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False)

        # Get information about the dataframe
        num_atts = len(X.iloc[0, 0])
        num_insts = X.shape[0]
        col_names = X.columns

        self.check_parameters(num_atts)

        df = pd.DataFrame()

        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = tabularize(pd.DataFrame(X[x]), return_array=True)

            # Calculate gradients
            transformedData = []
            for y in range(num_insts):
                res = self.get_gradients_of_lines(arr[y])
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

    """
    Function to get the gradients of the line of best fits given a time series.

    Parameters
    ----------
    X : a numpy array of shape = [num_atts]

    Returns
    -------
    m : a numpy array of shape = [num_intervals].
        It contains the gradients of the line of best fit
        for each interval in a time series.
    """
    def get_gradients_of_lines(self, X):
        # Firstly, split the time series into approx equal length intervals
        splitTimeSeries = self.split_time_series(X)
        gradients = []

        for x in range(len(splitTimeSeries)):
            gradients.append(self.get_gradient(splitTimeSeries[x]))

        return gradients

    """
    Function to get the gradient of the line of best fit given a
    section of a time series.

    Equation adopted from: real-statistics.com/regression/total-least-squares
    Parameters
    ----------
    Y : a numpy array of shape = [interval_size]

    Returns
    -------
    m : an int corresponding to the gradient of the best fit line.
    """
    def get_gradient(self, Y):
        # Create a list of content 1,2,3,4,...,len(Y) for the x coordinates.
        X = [(i+1) for i in range(len(Y))]

        # Calculate the mean of both lists
        meanX = statistics.mean(X)
        meanY = statistics.mean(Y)

        # Calculate the list (yi-mean(y))^2
        yminYbar = [(y-meanY)**2 for y in Y]
        # Calculate the list (xi-mean(x))^2
        xminXbar = [(x-meanX)**2 for x in X]

        # Sum them to produce w.
        w = sum(yminYbar) - sum(xminXbar)

        # Calculate the list (xi-mean(x))*(yi-mean(y))
        temp = []
        for x in range(len(X)):
            temp.append((X[x]-meanX)*(Y[x]-meanY))

        # Sum it and multiply by 2 to calculate r
        r = 2*sum(temp)

        if r == 0:
            # remove nans
            m = 0
        else:
            # Gradient is defined as (w+sqrt(w^2+r^2))/r
            m = (w+math.sqrt(w**2+r**2))/r

        return m

    """
    Function to split a time series into approximately equal intervals.

    Adopted from = https://stackoverflow.com/questions/2130016/
                   splitting-a-list-into-n-parts-of-approximately
                   -equal-length

    Parameters
    ----------
    X : a numpy array corresponding to the time series being split
        into approx equal length intervals.
    """
    def split_time_series(self, X):
        avg = len(X) / float(self.num_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning):int(beginning + avg)])
            beginning += avg

        return output

    """
    Function for checking the values of parameters inserted into Slope.

    Throws
    ------
    ValueError if a parameters input is invalid.
    """
    def check_parameters(self, num_atts):
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError("num_intervals must have the value \
                                  of at least 1")
            if self.num_intervals > num_atts:
                raise ValueError("num_intervals cannot be higher than \
                                  subsequence_length")
        else:
            raise ValueError("num_intervals must be an 'int'. Found '"
                             + type(self.num_intervals).__name__ + "' \
                             instead.")
