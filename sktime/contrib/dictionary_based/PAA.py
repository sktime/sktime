import numpy as np
import pandas as pd

from sktime.transformers.base import BaseTransformer


class PAA(BaseTransformer):
    __author__ = "Matthew Middlehurst"
    """ (PAA) Piecewise Aggregate Approximation Transformer, as described in 

    Overview: for each series: 
        run a sliding window accross the series
        for each window
            shorten the series with PAA (Piecewise Approximate Aggregation    
            """

    def __init__(self,
                 num_intervals=8,
                 dim_to_use=0
                 ):
        self.num_intervals = num_intervals

        self.dim_to_use = dim_to_use

        self.num_insts = 0
        self.num_atts = 0

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing "
                                "Series objects")

        self.num_atts = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("The fit method must be called before calling transform")

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing "
                                "Series objects")

        self.num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        for i in range(self.num_insts):
            series = X[i, :]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = self.num_atts / self.num_intervals
            frame_sum = 0

            for n in range(self.num_atts):
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

        dims['dim_' + str(self.dim_to_use)] = data

        return dims
