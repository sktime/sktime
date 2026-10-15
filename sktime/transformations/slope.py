"""Slope transformer."""

import numpy as np
import pandas as pd

from sktime.datatypes import convert
from sktime.transformations.base import BaseTransformer

__all__ = ["SlopeTransformer"]
__author__ = ["mloning", "AyushAnand413"]


class SlopeTransformer(BaseTransformer):
    """Slope-by-segment transformation.

    Class to perform the Slope transformation on a time series
    dataframe. It splits a time series into num_intervals segments.
    Then within each segment, it performs a total least
    squares regression to extract the gradient of the segment.

    Parameters
    ----------
    num_intervals : int, default=8
        Number of approximately equal segments to split the time series into.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.transformations.slope import SlopeTransformer
    >>> from sktime.datatypes import convert
    >>> X_3d = np.random.RandomState(42).normal(size=(5, 1, 20))
    >>> X = convert(X_3d, from_type="numpy3D", to_type="nested_univ")
    >>> transformer = SlopeTransformer(num_intervals=4)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (5, 1)
    """

    _tags = {
        "authors": ["mloning", "AyushAnand413"],
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
        "capability:categorical_in_X": False,
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
        avg = n_timepoints / float(self.num_intervals)

        for x in col_names:
            # Convert one of the columns in the dataframe to numpy array
            arr = convert(
                pd.DataFrame(X[x]),
                from_type="nested_univ",
                to_type="numpyflat",
                as_scitype="Panel",
            )

            # Vectorized gradient calculation across all instances simultaneously
            beginning = 0.0
            gradients = []
            while beginning < n_timepoints:
                start = int(beginning)
                end = int(beginning + avg)
                seg = arr[:, start:end]
                seg_len = seg.shape[1]

                if seg_len <= 1:
                    m = np.zeros(num_instances, dtype=np.float64)
                else:
                    x_coord = np.arange(1, seg_len + 1, dtype=np.float64)
                    mean_x = (seg_len + 1.0) / 2.0
                    x_dev = x_coord - mean_x
                    sum_xx = np.sum(x_dev**2)

                    mean_y = np.mean(seg, axis=1, keepdims=True)
                    y_dev = seg - mean_y
                    sum_yy = np.sum(y_dev**2, axis=1)

                    w = sum_yy - sum_xx
                    r = 2.0 * np.dot(y_dev, x_dev)

                    zero_r = r == 0.0
                    r_safe = np.where(zero_r, 1.0, r)
                    radicand = np.maximum(w * w + r * r, 0.0)
                    m = np.where(zero_r, 0.0, (w + np.sqrt(radicand)) / r_safe)

                gradients.append(m)
                beginning += avg

            transformedData = np.column_stack(gradients)
            Xt[x] = [pd.Series(transformedData[i]) for i in range(num_instances)]

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
        gradients : list of float
            It contains the gradients of the line of best fit
            for each interval in a time series.
        """
        splitTimeSeries = self._split_time_series(X)
        return [
            self._get_gradient(splitTimeSeries[x]) for x in range(len(splitTimeSeries))
        ]

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
        m : a float corresponding to the gradient of the best fit line.
        """
        Y = np.asarray(Y, dtype=np.float64)
        seg_len = len(Y)
        if seg_len <= 1:
            return 0.0

        X = np.arange(1, seg_len + 1, dtype=np.float64)
        mean_x = (seg_len + 1.0) / 2.0
        mean_y = np.mean(Y)

        x_dev = X - mean_x
        y_dev = Y - mean_y

        w = np.sum(y_dev**2) - np.sum(x_dev**2)
        r = 2.0 * np.dot(x_dev, y_dev)

        if r == 0.0:
            return 0.0

        return float((w + np.sqrt(w**2 + r**2)) / r)

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [{"num_intervals": 2}, {"num_intervals": 3}]
