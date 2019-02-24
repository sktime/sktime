from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from ..utils.validation import check_equal_index
from .series_to_series import RandomIntervalSegmenter

__all__ = ["RandomIntervalFeatureExtractor"]


class RandomIntervalFeatureExtractor(RandomIntervalSegmenter):
    """
    Splits time-series into random intervals and extracts features from each interval.
    Series-to-tabular transformer.
    """

    def __init__(self, n_intervals='sqrt', features=None, random_state=None, check_input=True):
        """
        Creates instance of RandomIntervalFeatureExtractor transformer.

        :param n_intervals: str or int
            - If "fixed", sqrt of length of time-series is used.
            - If "random", random number of intervals is generated.
            - If integer, integer gives (fixed) number of intervals to generate.

            Default is "sqrt".
        :param features: None or list of functions
            - If list of function, applies each function to random intervals to extract features.
            - If None, the mean is used.

            Default is None.

        :param random_state:
        :param check_input:
        """
        super(RandomIntervalFeatureExtractor, self).__init__(
            n_intervals=n_intervals,
            random_state=random_state,
            check_input=check_input
        )
        self.feature_names_ = []

        # Check input of feature calculators, i.e list of functions to be applied to time-series
        if features is None:
            self.features = [np.mean]
        elif isinstance(features, list) and all([callable(func) for func in features]):
            self.features = features
        else:
            raise ValueError('Features must be list containing only functions (callable) to be '
                             'applied to the data columns')

    def transform(self, X, y=None):
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

        # Convert into 3d numpy array, only possible for equal-index time-series data.
        Xarr = np.array([np.array([row for row in X.iloc[:, col].tolist()])
                         for col, _ in enumerate(X.columns)])

        # Pre-allocate arrays.
        n_rows, n_cols = X.shape
        n_features = len(self.features)
        n_cols_intervals = sum([intervals.shape[0] for intervals in self.intervals_])  # total number of intervals of all columns
        Xt = np.zeros((n_rows, n_features * n_cols_intervals))  # Allocate output array for transformed data
        feature_names_ = []

        # Compute features on intervals.
        i = 0
        for c, col in enumerate(X.columns):
            for start, end in self.intervals_[c]:
                interval = Xarr[c, :, start:end]
                for func in self.features:
                    Xt[:, i] = np.apply_along_axis(func, 1, interval)
                    i += 1
                    feature_names_.append(f'{col}_{start}_{end}_{func.__name__}')

        self.feature_names_ = feature_names_
        return pd.DataFrame(Xt, columns=self.feature_names_)


# class FeatureExtractor(BaseTransformer):
#     """
#     Series-to-tabular transformer.
#     """
#     def __init__(self, feature_calculators=None):
#         self.input_shape_ = None
#         self.input_indexes_ = []  # list of time-series indexes of each column
#
#         # Check input of feature calculators, i.e list of functions to be applied to time-series
#         if feature_calculators is None:
#             self.feature_calculators = [np.mean]
#         else:
#             if not isinstance(feature_calculators, list):
#                 if not all([callable(f) for f in feature_calculators]):
#                     raise ValueError('Features must be list containing only functions (callable) to be '
#                                      'applied to the data columns')
#             else:
#                 self.feature_calculators = feature_calculators
#
#     def fit(self, X, y=None):
#         self.input_shape_ = X.shape
#         self.input_indexes_ = check_equal_index(X)
#
#         # Return the transformer
#         return self
#
#     def transform(self, X):
#         """
#         Segment series into random intervals.
#         """
#         # Check is fit had been called
#         check_is_fitted(self, ['input_shape_'])
#
#         # Check that the input is of the same shape as the one passed
#         # during fit.
#         if X.shape[1] != self.input_shape_[1]:
#             raise ValueError('Number of columns of input is different from what was seen'
#                              'in `fit`')
#         # Input validation
#         if not all([fit_idx.equals(trans_idx) for trans_idx, fit_idx in zip(check_equal_index(X),
#                                                                             self.input_indexes_)]):
#             raise ValueError('Indexes of time-series are different from what was seen in `fit`')
#
#         # Transform input data
#         n_rows, n_cols = X.shape
#
#         calculated_features_dict = {}
#         for calculator in self.feature_calculators:
#             for col in range(n_cols):
#                 col_name = f'{X.columns[col]}_{calculator.__name__}'
#                 calculated_features_dict[col_name] = X.iloc[:, col].apply(calculator)
#
#         return pd.DataFrame(calculated_features_dict)

