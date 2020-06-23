
# import numpy as np
import pandas as pd
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.data_container import detabularize
# from sktime.utils.data_container import tabularize
import numpy as np

__all__ = ["TruncationTransformer"]
__author__ = ["Aaron Bostrom"]


class TruncationTransformer(BaseSeriesAsFeaturesTransformer):
    """MyTransformer docstring
    """

    def __init__(self, lower=None, upper=None, dim_to_use=0):
        self.lower = lower
        self.upper = upper
        self.dim_to_use = dim_to_use
        super(TruncationTransformer, self).__init__()

    def fit(self, X, y=None):
        """
        Fit transformer.

        Parameters
        ----------
        X : pandas DataFrame of shape [n_samples, n_features]
            Input data
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : an instance of self.
        """
        X = check_X(X, enforce_univariate=True)

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Transform X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_columns]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
        """
        # X = check_X(X, enforce_univariate=True)

        # Tabularise assuming series
        # arr = tabularize(X, return_array=True)

        n_instances, n_timepoints = X.shape

        arr = [X.iloc[i, :].values for i in range(n_instances)]

        def get_length(input):
            return len(input[self.dim_to_use])

        # depending on inputs either find the shortest truncation.
        # or use the bounds.
        if self.lower is None:
            idxs = np.arange(min(map(get_length, arr)))
        else:
            if self.upper is None:
                idxs = np.arange(self.lower)
            else:
                idxs = np.arange(self.lower, self.upper)

        truncate = [out[self.dim_to_use][idxs] for out in arr]

        # truncate between lower and upper inclusive.
        # truncate = arr[:, self.lower:self.upper+1]

        # retabularize
        return detabularize(pd.DataFrame(truncate))
