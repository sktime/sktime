from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from ..utils.validation import check_equal_index
from ..utils.transformations import tabularize, concat_nested_arrays
from ..utils.time_series import rand_intervals_rand_n, rand_intervals_fixed_n
from .base import BaseTransformer
from sklearn.preprocessing import FunctionTransformer

__all__ = ['RandomIntervalSegmenter', 'DerivativeSlopeTransformer']


class RandomIntervalSegmenter(BaseTransformer):
    """Transformer that segments time-series into random intervals.

    Parameters
    ----------

    param n_intervals: str or int
        Number of intervals to generate.
        - If "sqrt", sqrt of length of time-series is used.
        - If "random", random number of intervals is generated.
        - If int, n_intervals intervals are generated.
        Default is "sqrt".

    param random_state: : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    param check_input: bool, optional (default=True)
        When set to ``True``, inputs will be validated, otherwise inputs are assumed to be valid
        and no checks are performed. Use with caution.
    """

    def __init__(self, n_intervals='sqrt', min_length=None, random_state=None, check_input=True):


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
        """Fit transformer, generating random interval indices.

        Parameters
        ----------
        X : pandas DataFrame of shape [n_samples, n_features]
            Input data
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : RandomIntervalSegmenter
            This estimator
        """
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
        """Transform X, segments time-series in each column into random intervals using interval indices generated
        during `fit`.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
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


class DerivativeSlopeTransformer(BaseTransformer):

    def transform(self, X, y=None):
        num_cases, num_dim = X.shape
        output_df = pd.DataFrame()
        for dim in range(num_dim):
            dim_data = X.iloc[:,dim]
            out = DerivativeSlopeTransformer.row_wise_get_der(dim_data)
            output_df['der_dim_'+str(dim)] = pd.Series(out)

        return output_df

    @staticmethod
    def row_wise_get_der(X):

        def get_der(x):
            der = []
            for i in range(1, len(x) - 1):
                der.append(((x[i] - x[i - 1]) + ((x[i + 1] - x[i - 1]) / 2)) / 2)
            return pd.Series([der[0]] + der + [der[-1]])

        return [get_der(x) for x in X]
