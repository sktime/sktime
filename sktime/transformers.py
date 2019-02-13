from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_random_state, check_is_fitted
import numpy as np
import pandas as pd
from .utils.validation import check_equal_index


__all__ = ["RandomIntervalSegmenter", "FeatureExtractor"]


class RandomIntervalSegmenter(BaseEstimator, TransformerMixin):
    """
    Series-to-series transformer.
    """
    def __init__(self, random_state=None):
        self.input_shape_ = None
        self.input_indexes_ = []  # list of time-series indexes of each column
        self.intervals_ = []  # list of random intervals of each column
        self.random_state = random_state

    def fit(self, X, y=None):
        self.input_shape_ = X.shape
        self.input_indexes_ = check_equal_index(X)

        # Compute random intervals
        def _random_intervals(index, random_state=self.random_state):
            """
            Obtain random intervals from index.
            """
            rng = check_random_state(random_state)

            def _random_choice(x, size):
                return rng.choice(x, size=size, replace=False)

            starts = []
            ends = []
            m = index.size  # series length
            idx = np.arange(1, m + 1)

            W = _random_choice(idx, size=int(np.sqrt(m)))
            for w in W:
                size = m - w + 1
                start = _random_choice(np.arange(1, size + 1),
                                       size=int(np.sqrt(size))) - 1
                starts.extend(start)
                for s in start:
                    end = s + w
                    ends.append(end)
            return starts, ends

        n_cols = self.input_shape_[1]
        for col in range(n_cols):
            starts, ends = _random_intervals(self.input_indexes_[col])
            self.intervals_.append(np.column_stack([starts, ends]))

        # Return the transformer
        return self

    def transform(self, X, y=None):
        """
        Segment series into random intervals. Series-to-series transformer.
        """

        # Check is fit had been called
        check_is_fitted(self, ['input_shape_', 'intervals_'])

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns of input is different from what was seen'
                             'in `fit`')
        # Input validation
        if not all([fit_idx.equals(trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
                                                                            self.input_indexes_)]):
            raise ValueError('Indexes of time-series are different from what was seen in `fit`')

        # Transform input data using random intervals from `fit`
        n_rows, n_cols = X.shape
        interval_data_dict = {}
        for col in range(n_cols):
            col_name = X.columns[col]
            for start, end in self.intervals_[col]:
                interval_data_list = []
                for row in range(n_rows):
                    interval_data = X.iloc[row, col].iloc[start:end]
                    interval_data_list.append(interval_data)
                interval_data_dict[f'{col_name}_{start}_{end}'] = interval_data_list

        return pd.DataFrame(interval_data_dict)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Series-to-tabular transformer.
    """
    def __init__(self, feature_calculators=None):
        self.input_shape_ = None
        self.input_indexes_ = []  # list of time-series indexes of each column

        # Check input of feature calculators, i.e list of functions to be applied to time-series
        if feature_calculators is None:
            self.feature_calculators = [np.mean]
        else:
            if not isinstance(feature_calculators, list):
                if not all([callable(f) for f in feature_calculators]):
                    raise ValueError('Features must be list containing only functions (callable) to be '
                                     'applied to the data columns')
            else:
                self.feature_calculators = feature_calculators

    def fit(self, X, y=None):
        self.input_shape_ = X.shape
        self.input_indexes_ = check_equal_index(X)

        # Return the transformer
        return self

    def transform(self, X):
        """
        Segment series into random intervals.
        """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns of input is different from what was seen'
                             'in `fit`')
        # Input validation
        if not all([fit_idx.equals(trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
                                                                            self.input_indexes_)]):
            raise ValueError('Indexes of time-series are different from what was seen in `fit`')

        # Transform input data
        n_rows, n_cols = X.shape

        calculated_features_dict = {}
        for calculator in self.feature_calculators:
            for col in range(n_cols):
                col_name = f'{X.columns[col]}_{calculator.__name__}'
                calculated_features_dict[col_name] = X.iloc[:, col].apply(calculator)

        return pd.DataFrame(calculated_features_dict)

