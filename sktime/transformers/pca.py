import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA

from sktime.transformers.base import BaseTransformer
from sktime.utils.validation.supervised import validate_X, check_X_is_univariate
from sktime.utils.data_container import tabularise, detabularise, check_equal_index


class PCATransformer(BaseTransformer):
    """ Transformer that applies Principle Components Analysis to a univariate time series.

    Provides a simple wrapper around ``sklearn.decomposition.PCA``.

    Parameters
    ----------
    n_components : int, float, str or None (default None)
        Number of principle components to retain. By default, all components are retained. See
        ``sklearn.decomposition.PCA`` documentation for a detailed description of all options.
    **kwargs
        Additional parameters passed on to ``sklearn.decomposition.PCA``. See ``sklearn.decomposition.PCA``
        documentation for a detailed description of all options.
    """

    def __init__(self, n_components=None, **kwargs):
        self.pca = PCA(n_components, **kwargs)

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

        validate_X(X)
        check_X_is_univariate(X)

        # Transform the time series column into tabular format and
        # apply PCA to the tabular format
        Xtab = tabularise(X)
        self.pca.fit(Xtab)

        return self

    def transform(self, X, y=None):
        """
        Transform X, transforms univariate time-series using sklearn's PCA class

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_samples, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
          Transformed pandas DataFrame with the same number of rows and the (potentially reduced) PCA transformed
          column. Time indices of the original column are replaced with 0:(n_components - 1).
        """

        # Check inputs.
        check_is_fitted(self.pca, 'n_components_')
        validate_X(X)
        check_X_is_univariate(X)

        # Transform X using the fitted PCA
        Xtab = tabularise(X)
        Xpca = pd.DataFrame(data=self.pca.transform(Xtab),
                            index=Xtab.index,
                            columns=Xtab.columns[:self.pca.n_components_])

        # Back-transform into time series data format
        Xt = detabularise(Xpca, index=X.index)
        Xt.columns = X.columns

        return Xt