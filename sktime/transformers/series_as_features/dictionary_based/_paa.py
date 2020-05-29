import pandas as pd
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X

__author__ = "Matthew Middlehurst"


class PAA(BaseSeriesAsFeaturesTransformer):
    """ (PAA) Piecewise Aggregate Approximation Transformer, as described in
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

    def __init__(self,
                 num_intervals=8
                 ):
        self.num_intervals = num_intervals
        super(PAA, self).__init__()

    def set_num_intervals(self, n):
        self.num_intervals = n

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)

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
                    current_frame_size = (1 - remaining)

            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
                frames.append(frame_sum / frame_length)

            data.append(pd.Series(frames))

        dims[0] = data

        return dims
