""" KMeans time series clustering built on sklearn KMeans that
supports a range of distance measure specifically for time series. This distance
functions are defined in cython in sktime.distances.elastic_cython. Python versions
are in sktime.distances.elastic, but these are orders of magnitude slower.

"""

__author__ = "Josue Davalos"
__all__ = ["KMeamsTimeSeries"]

from sklearn.cluster import KMeans as _KMeans
from sktime.utils.validation.panel import check_X
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin 


class KMeamsTimeSeries(_KMeans, ClusterMixin):
    """
    An adapted version of the scikit-learn KMeans to work with
    time series data.

    Necessary changes required for time series data:
        -   This assumes that data must be 2d. 


    Parameters
    ----------
    n_clusters     : int, set k for kmeans (default =1)
    """

    def __init__(
        self,
        n_clusters=1,
        metric="euclidean",
        distance_params=None,
        **kwargs
    ):

        super(KMeamsTimeSeries, self).__init__(
            n_clusters = n_clusters,
            **kwargs
        )
        self._is_fitted = False


    def fit(self, X, y=None):
        """Fit the model using X as training data 

        Parameters
        ----------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])
        """
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        # Reshape to work correctly
        X = X_reshape(X)
        fx = super().fit(X)

        self._is_fitted = True
        return fx
    

    def predict(self, X):
        """Predict the cluster labels for the provided data

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n_query,
        n_features)

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Cluster labels for each data sample.
        """
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)
        X = X_reshape(X)
        y_pred = super().predict(X)
        return y_pred


def X_reshape(X):
    nsamples, nx, ny = X.shape
    return X.reshape((nsamples,nx*ny))
