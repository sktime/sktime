__author__ = ["Patrick Rockenschaub"]
__all__ = ["PCATransformer"]

import pandas as pd
from sklearn.decomposition import PCA
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.utils.data_container import detabularise
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X


class PCATransformer(BaseSeriesAsFeaturesTransformer):
    """ Transformer that applies Principle Components Analysis to a
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
        X = check_X(X, enforce_univariate=True)

        # Transform the time series column into tabular format and
        # apply PCA to the tabular format
        Xtab = tabularize(X)
        self.pca.fit(Xtab)
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
        X = check_X(X, enforce_univariate=True)

        # Transform X using the fitted PCA
        Xtab = tabularize(X)
        Xpca = pd.DataFrame(data=self.pca.transform(Xtab),
                            index=Xtab.index,
                            columns=Xtab.columns[:self.pca.n_components_])

        # Back-transform into time series data format
        Xt = detabularise(Xpca, index=X.index)
        Xt.columns = X.columns
        return Xt
