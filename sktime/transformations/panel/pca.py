# -*- coding: utf-8 -*-
__author__ = ["Patrick Rockenschaub"]
__all__ = ["PCATransformer"]

import pandas as pd
from sklearn.decomposition import PCA

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.utils.validation.panel import check_X


class PCATransformer(_PanelToPanelTransformer):
    """Transformer that applies Principle Components Analysis to a
    univariate time series.

    Provides a simple wrapper around ``sklearn.decomposition.PCA``.

    Parameters
    ----------
    n_components : int, float, str or None (default None)
        Number of principle components to retain. By default, all components
        are retained. See
        ``sklearn.decomposition.PCA`` documentation for a detailed
        description of all options.
    **kwargs
        Additional parameters passed on to ``sklearn.decomposition.PCA``.
        See ``sklearn.decomposition.PCA``
        documentation for a detailed description of all options.
    """

    _tags = {"univariate-only": True}

    def __init__(self, n_components=None, **kwargs):
        self.n_components = n_components
        self.pca = PCA(self.n_components, **kwargs)
        super(PCATransformer, self).__init__()

    def fit(self, X, y=None):
        """
        Fit transformer, finding all principal components.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        self : an instance of self.
        """
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        # Transform the time series column into tabular format and
        # apply PCA to the tabular format
        self.pca.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Transform X, transforms univariate time-series using sklearn's PCA
        class

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with the same number of rows and the
          (potentially reduced) PCA transformed
          column. Time indices of the original column are replaced with 0:(
          n_components - 1).
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        # Transform X using the fitted PCA
        Xpca = pd.DataFrame(data=self.pca.transform(X))

        # Back-transform into time series data format
        Xt = from_2d_array_to_nested(Xpca)
        return Xt
