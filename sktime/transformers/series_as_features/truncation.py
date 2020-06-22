
import numpy as np
import pandas as pd
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X
from sktime.datasets.base import load_gunpoint
from sktime.utils.data_container import tabularize, detabularize
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

__all__ = ["TruncationTransformer"]
__author__ = ["Aaron Bostrom"]


class TruncationTransformer(BaseSeriesAsFeaturesTransformer):
    """MyTransformer docstring
    """

    def __init__(self, lower, upper, dim_to_use=0):
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
        #X = check_X(X, enforce_univariate=True)

        # Tabularise assuming series
        arr = tabularize(X, return_array=True)

        #truncate between lower and upper inclusive.
        truncate = arr[:, self.lower:self.upper+1]
        
        #retabularize
        return detabularize(pd.DataFrame(truncate))


