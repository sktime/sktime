from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from ..utils.validation import check_equal_index
from ..utils.transformations import tabularize, concat_nested_arrays
from ..utils.time_series import rand_intervals_rand_n, rand_intervals_fixed_n
from .base import BaseTransformer

__all__ = ['RandomIntervalSegmenter']


class RandomIntervalSegmenter(BaseTransformer):
    """
    Series-to-series transformer.
    """

    def __init__(self, n_intervals='sqrt', min_length=None, random_state=None, check_input=True):
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
        self.columns_ = []
        if min_length is None:
            self.min_length = 1
        else:
            self.min_length = min_length

        if n_intervals in ('sqrt', 'random'):
            self.n_intervals = n_intervals
        elif np.issubdtype(type(n_intervals), np.integer):
            if n_intervals <= 0:
                raise ValueError('Number of intervals must be positive')
            self.n_intervals = n_intervals
        else:
            raise ValueError(f'Number of intervals must be either "random", "sqrt" or positive integer, '
                             f'but found {type(n_intervals)}')

    def fit(self, X, y=None):
        if self.check_input:
            # TODO check input is series column, not column of primitives
            pass

        # Cast into 2d dataframe
        if X.ndim == 1:
            X = pd.DataFrame(X)

        self.input_shape_ = X.shape

        # Retrieve time-series indexes from each column.
        # TODO generalise to columns with series of unequal length
        self.input_indexes_ = [X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index')
                               else np.arange(X.iloc[0, c].shape[0]) for c in range(self.input_shape_[1])]

        # Compute random intervals for each column.
        # TODO if multiple columns are passed, introduce option to compute one set of shared intervals,
        #  or use ColumnTransformer?
        if self.n_intervals == 'random':
            self.intervals_ = [rand_intervals_rand_n(self.input_indexes_[c],
                                                     random_state=self.random_state)
                               for c in range(self.input_shape_[1])]
        else:
            self.intervals_ = [rand_intervals_fixed_n(self.input_indexes_[c],
                                                      n=self.n_intervals,
                                                      min_length=self.min_length,
                                                      random_state=self.random_state)
                               for c in range(self.input_shape_[1])]
        return self

    def transform(self, X, y=None):
        """
        Segment series into random intervals. Series-to-series transformer.
        """

        # Check is fit had been called
        check_is_fitted(self, 'intervals_')

        # Cast into 2d dataframe
        if X.ndim == 1:
            X = pd.DataFrame(X)

        # Check inputs.
        if self.check_input:
            # Check that the input is of the same shape as the one passed
            # during fit.
            if (X.shape[1] if X.ndim == 2 else 1) != self.input_shape_[1]:
                raise ValueError('Number of columns of input is different from what was seen'
                                 'in `fit`')
            # Input validation
            if not all([np.array_equal(fit_idx, trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
                                                                                         self.input_indexes_)]):
                raise ValueError('Indexes of input time-series are different from what was seen in `fit`')

        # Segment into intervals.
        intervals = []
        self.columns_ = []
        for c, (colname, col) in enumerate(X.items()):
            # Tabularize each column assuming series have equal indexes in any given column.
            # TODO generalise to non-equal-index cases
            arr = tabularize(col, return_array=True)
            for start, end in self.intervals_[c]:
                interval = arr[:, start:end]
                intervals.append(interval)
                self.columns_.append(f'{colname}_{start}_{end}')

        # Return nested pandas Series or DataFrame.
        Xt = pd.DataFrame(concat_nested_arrays(intervals, return_arrays=True))
        Xt.columns = self.columns_
        return Xt
