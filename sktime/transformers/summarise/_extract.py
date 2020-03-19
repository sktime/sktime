#!/usr/bin/env python3 -u
# coding: utf-8

import numpy as np
import pandas as pd
from sktime.utils.validation import check_is_fitted
from sktime.transformers.base import BaseTransformer
from sktime.transformers.segment import RandomIntervalSegmenter
from sktime.utils.data_container import tabularize
from sktime.utils.validation.supervised import validate_X, check_X_is_univariate


class PlateauFinder(BaseTransformer):
    """Transformer that finds segments of the same given value, plateau in the time series, and
    returns the starting indices and lengths.

    Parameters
    ----------
    value : {int, float, np.nan, np.inf}
        Value for which to find segments
    min_length : int
        Minimum lengths of segments with same value to include.
        If min_length is set to 1, the transformer can be used as a value finder.
    """

    def __init__(self, value=np.nan, min_length=2):
        self.value = value
        self.min_length = min_length

        self._starts = []
        self._lengths = []

    def transform(self, X, y=None):
        """Transform X.
        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, n_columns]
            Nested dataframe with time-series in cells.
        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame
        """

        # input checks
        validate_X(X)
        check_X_is_univariate(X)

        # get column name
        column_name = X.columns[0]

        # find plateaus (segments of the same value)
        for x in X.iloc[:, 0]:
            x = np.asarray(x)

            # find indices of transition
            if np.isnan(self.value):
                i = np.where(np.isnan(x), 1, 0)

            elif np.isinf(self.value):
                i = np.where(np.isinf(x), 1, 0)

            else:
                i = np.where(x == self.value, 1, 0)

            # pad and find where segments transition
            transitions = np.diff(np.hstack([0, i, 0]))

            # compute starts, ends and lengths of the segments
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            lengths = ends - starts

            # filter out single points
            starts = starts[lengths >= self.min_length]
            lengths = lengths[lengths >= self.min_length]

            self._starts.append(starts)
            self._lengths.append(lengths)

        # put into dataframe
        Xt = pd.DataFrame()
        column_prefix = "%s_%s" % (column_name, "nan" if np.isnan(self.value) else str(self.value))
        Xt["%s_starts" % column_prefix] = pd.Series(self._starts)
        Xt["%s_lengths" % column_prefix] = pd.Series(self._lengths)
        return Xt


class DerivativeSlopeTransformer(BaseTransformer):
    # TODO add docstrings
    def transform(self, X, y=None):
        num_cases, num_dim = X.shape
        output_df = pd.DataFrame()
        for dim in range(num_dim):
            dim_data = X.iloc[:, dim]
            out = DerivativeSlopeTransformer.row_wise_get_der(dim_data)
            output_df['der_dim_' + str(dim)] = pd.Series(out)

        return output_df

    @staticmethod
    def row_wise_get_der(X):

        def get_der(x):
            der = []
            for i in range(1, len(x) - 1):
                der.append(((x[i] - x[i - 1]) + ((x[i + 1] - x[i - 1]) / 2)) / 2)
            return pd.Series([der[0]] + der + [der[-1]])

        return [get_der(x) for x in X]


class RandomIntervalFeatureExtractor(RandomIntervalSegmenter):
    """
    Transformer that segments time-series into random intervals
    and subsequently extracts series-to-primitives features from each interval.

    n_intervals: str{'sqrt', 'log', 'random'}, int or float, optional (default='sqrt')
        Number of random intervals to generate, where m is length of time series:
        - If "log", log of m is used.
        - If "sqrt", sqrt of m is used.
        - If "random", random number of intervals is generated.
        - If int, n_intervals intervals are generated.
        - If float, int(n_intervals * m) is used with n_intervals giving the fraction of intervals of the
        time series length.

        For all arguments relative to the length of the time series, the generated number of intervals is
        always at least 1.

    features: list of functions, optional (default=None)
        Applies each function to random intervals to extract features.
        If None, only the mean is extracted.

    random_state: : int, RandomState instance, optional (default=None)
        - If int, random_state is the seed used by the random number generator;
        - If RandomState instance, random_state is the random number generator;
        - If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, n_intervals='sqrt', min_length=2, features=None, random_state=None):
        super(RandomIntervalFeatureExtractor, self).__init__(
            n_intervals=n_intervals,
            min_length=min_length,
            random_state=random_state,
        )

        # Check input of feature calculators, i.e list of functions to be applied to time-series
        if features is None:
            self.features = [np.mean]
        elif isinstance(features, list) and all([callable(func) for func in features]):
            self.features = features
        else:
            raise ValueError('Features must be list containing only functions (callables) to be '
                             'applied to the data columns')

    def transform(self, X, y=None):
        """
        Transform X, segments time-series in each column into random intervals using interval indices generated
        during `fit` and extracts features from each interval.

        Parameters
        ----------
        X : nested pandas.DataFrame of shape [n_samples, n_features]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas.DataFrame
          Transformed pandas DataFrame with same number of rows and one column for each generated interval.
        """
        # Check is fit had been called
        check_is_fitted(self, 'intervals_')
        validate_X(X)
        check_X_is_univariate(X)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns of input is different from what was seen'
                             'in `fit`')
        # Input validation
        # if not all([np.array_equal(fit_idx, trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
        #                                                                              self._time_index)]):
        #     raise ValueError('Indexes of input time-series are different from what was seen in `fit`')

        n_rows, n_columns = X.shape
        n_features = len(self.features)

        n_intervals = len(self.intervals_)

        # Compute features on intervals.
        Xt = np.zeros((n_rows, n_features * n_intervals))  # Allocate output array for transformed data
        self.columns_ = []
        colname = X.columns[0]

        # Tabularize each column assuming series have equal indexes in any given column.
        # TODO generalise to non-equal-index cases
        arr = tabularize(X, return_array=True)
        i = 0
        for func in self.features:
            # TODO generalise to series-to-series functions and function kwargs
            for start, end in self.intervals_:
                interval = arr[:, start:end]

                # Try to use optimised computations over axis if possible, otherwise iterate over rows.
                try:
                    Xt[:, i] = func(interval, axis=1)
                except TypeError as e:
                    if str(e) == f"{func.__name__}() got an unexpected keyword argument 'axis'":
                        Xt[:, i] = np.apply_along_axis(func, 1, interval)
                    else:
                        raise
                i += 1
                self.columns_.append(f'{colname}_{start}_{end}_{func.__name__}')

        Xt = pd.DataFrame(Xt)
        Xt.columns = self.columns_
        return Xt
