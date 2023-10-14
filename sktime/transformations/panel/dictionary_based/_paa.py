"""Piecewise Aggregate Approximation Transformer (PAA)."""
import pandas as pd

from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.transformations.base import BaseTransformer

__author__ = ["MatthewMiddlehurst"]


class PAA(BaseTransformer):
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
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
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
        """Transform data.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_dims]
            Nested dataframe with multivariate time-series in cells.

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero,
              second in column one etc.
        """
        # Get information about the dataframe
        num_atts = len(X.iloc[0, 0])
        col_names = X.columns

        # Check the parameters are appropriate
        self._check_parameters(num_atts)

        # On each dimension, perform PAA
        dataFrames = []
        for x in col_names:
            dataFrames.append(self._perform_paa_along_dim(pd.DataFrame(X[x])))

        # Combine the dimensions together
        result = pd.concat(dataFrames, axis=1, sort=False)
        result.columns = col_names

        return result

    def _perform_paa_along_dim(self, X):
        X = from_nested_to_2d_array(X, return_numpy=True)

        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        for i in range(num_insts):
            series = X[i, :]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            for n in range(num_atts):
                remaining = frame_length - current_frame_size

                if remaining > 1:
                    frame_sum += series[n]
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    current_frame_size += remaining

                if current_frame_size == frame_length:
                    frames.append(frame_sum / frame_length)
                    current_frame += 1

                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining

            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
                frames.append(frame_sum / frame_length)

            data.append(pd.Series(frames))

        dims[0] = data

        return dims

    def _check_parameters(self, num_atts):
        """Check parameters of PAA.

        Function for checking the values of parameters inserted into PAA.
        For example, the number of subsequences cannot be larger than the
        time series length.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError(
                    "num_intervals must have the \
                                  value of at least 1"
                )
            if self.num_intervals > num_atts:
                raise ValueError(
                    "num_intervals cannot be higher \
                                  than the time series length."
                )
        else:
            raise TypeError(
                "num_intervals must be an 'int'. Found '"
                + type(self.num_intervals).__name__
                + "' instead."
            )
