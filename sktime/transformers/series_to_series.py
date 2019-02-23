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

    def __init__(self, n_intervals='random', features=None, random_state=None, check_input=True):
        """
        Creates instance of RandomIntervalFeatureExtractor transformer.

        :param n_intervals: str or int
            If "fixed", sqrt of length of time-series is used. If "random", random number of intervals is generated.
            Use integer to specify (fixed) number of intervals to generate. Default is "random".
        :param features:
        :param random_state:
        :param check_input:
        """
        self.input_indexes_ = []  # list of time-series indexes of each column
        self.random_state = random_state
        self.check_input = check_input
        self.intervals_ = []
        self.input_shape_ = ()
        self.is_fitted_ = False

        # Check input of feature calculators, i.e list of functions to be applied to time-series
        if features is None:
            raise ValueError('Must supply a list of functions to extract features')
        else:
            if isinstance(features, list):
                if not all([callable(f) for f in features]):
                    raise ValueError('Features must be list containing only functions (callable) to be '
                                     'applied to the data columns')
                else:
                    self.features = features

        if n_intervals == 'fixed':
            self.n_intervals = None
        elif np.issubdtype(type(n_intervals), np.integer) or 'random':
            self.n_intervals = n_intervals
        else:
            raise ValueError('Number of intervals must be either "random", "fixed" or integer')

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
        self.is_fitted_ = True

        return self

    def transform(self, X, y=None):
        """
        Segment series into random intervals. Series-to-series transformer.
        """

        """
        Segment series into random intervals. Series-to-series transformer.
        """

        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

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

        n_rows, n_cols = X.shape
        n_features = len(self.features)
        n_intervals = sum([len(intervals) for intervals in self.intervals_])  # total number of intervals of all columns

        # Convert into 3d numpy array, only possible for equal-index time-series data
        Xarr = np.array([np.array([row for row in X.iloc[:, col].tolist()])
                         for col, _ in enumerate(X.columns)])

        # Compute features on intervals
        Xt = np.zeros((n_rows, n_intervals))  # Allocate output array for transformed data

        intervals_ = []
        for c, col in enumerate(X.columns):
            for i, (start, end) in enumerate(self.intervals_[c]):
                interval = Xarr[c, :, start:end]
                intervals_.append(interval)

        raise NotImplementedError

