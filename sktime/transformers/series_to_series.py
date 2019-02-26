from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from ..utils.validation import check_equal_index
from ..utils.time_series import rand_intervals_rand_n, rand_intervals_fixed_n
from .base import BaseTransformer

__all__ = ["RandomIntervalSegmenter"]


class RandomIntervalSegmenter(BaseTransformer):
    """
    Series-to-series transformer.
    """

    def __init__(self, n_intervals='sqrt', random_state=None, check_input=True):
        """
        Creates instance of RandomIntervalFeatureExtractor transformer.

        :param n_intervals: str or int
            - If "sqrt", sqrt of length of time-series is used.
            - If "random", random number of intervals is generated.
            - If int, n_intervals intervals are generated.

            Default is "sqrt".
        :param random_state:
        :param check_input:
        """
        self.input_indexes_ = []  # list of time-series indexes of each column
        self.random_state = random_state
        self.check_input = check_input
        self.intervals_ = []
        self.input_shape_ = ()
        self.n_intervals = n_intervals
        self.feature_names_ = []

        if n_intervals in ('sqrt', 'random'):
            self.n_intervals = n_intervals
        elif np.issubdtype(type(n_intervals), np.integer):
            if n_intervals == 0:
                raise ValueError('Number of intervals must be positive')
            self.n_intervals = n_intervals
        else:
            raise ValueError(f'Number of intervals must be either "random", "sqrt" or positive integer, '
                             f'but found {type(n_intervals)}')

    def fit(self, X, y=None):
        self.input_shape_ = X.shape

        if self.check_input:
            self.input_indexes_ = check_equal_index(X)
        else:
            self.input_indexes_ = [X.iloc[0, c].index for c in range(self.input_shape_[1])]

        # Compute random intervals for each column
        intervals_ = []
        if self.n_intervals == 'random':
            for c in range(self.input_shape_[1]):
                intervals = rand_intervals_rand_n(self.input_indexes_[c],
                                                  random_state=self.random_state)
                intervals_.append(intervals)
        else:
            for c in range(self.input_shape_[1]):
                intervals = rand_intervals_fixed_n(self.input_indexes_[c], n=self.n_intervals,
                                                   random_state=self.random_state)
                intervals_.append(intervals)

        self.intervals_ = intervals_

        return self

    def transform(self, X, y=None):
        """
        Segment series into random intervals. Series-to-series transformer.
        """

        """
        Segment series into random intervals. Series-to-series transformer.
        """

        # Check is fit had been called
        check_is_fitted(self, 'intervals_')

        # check inputs
        if self.check_input:
            # Check that the input is of the same shape as the one passed
            # during fit.
            if X.shape[1] != self.input_shape_[1]:
                raise ValueError('Number of columns of input is different from what was seen'
                                 'in `fit`')
            # Input validation
            if not all([fit_idx.equals(trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
                                                                                self.input_indexes_)]):
                raise ValueError('Indexes of input time-series are different from what was seen in `fit`')

        # Segment into intervals
        Xarr = np.array([np.array([row for row in X.iloc[:, col].tolist()])
                         for col, _ in enumerate(X.columns)])

        intervals = []
        for c, col in enumerate(X.columns):
            for start, end in self.intervals_[c]:
                interval = Xarr[c, :, start:end]
                intervals.append(interval)
                self.feature_names_.append(f'{col}_{start}_{end}')

        Xt = pd.DataFrame([pd.Series([pd.Series(row) for row in interval]) for interval in intervals]).T
        Xt.columns = self.feature_names_
        return Xt
