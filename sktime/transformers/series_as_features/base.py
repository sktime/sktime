from sktime.base import BaseEstimator

__all__ = ["BaseSeriesAsFeaturesTransformer",
           "is_series_as_features_transformer",
           "is_non_fittable_series_as_features_transformer"]
__author__ = ["Markus Löning", "Sajay Ganesh"]


class BaseSeriesAsFeaturesTransformer(BaseEstimator):
    """
    Base class for transformers, for identification.
    """

    def fit(self, X, y=None):
        """
        empty fit function, which inheriting transformers can override
        if need be.
        """
        self._is_fitted = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        # fit method of arity 2 (supervised transformation)
        return self.fit(X, y, **fit_params).transform(X)


def is_series_as_features_transformer(estimator):
    """Return True if the given estimator is (probably) a series-as-features
    transformer.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a series-as-features transformer and False
        otherwise.
    """
    return isinstance(estimator, BaseSeriesAsFeaturesTransformer)


class _NonFittableSeriesAsFeaturesTransformer(BaseSeriesAsFeaturesTransformer):
    """Base class for transformers which do nothing in fit and if fittable,
    fit during transform, otherwise only transform data"""
    pass


def is_non_fittable_series_as_features_transformer(estimator):
    """Return True if the given estimator is (probably) a series-as-features
    transformer.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a series-as-features transformer and False
        otherwise.
    """
    return isinstance(estimator, _NonFittableSeriesAsFeaturesTransformer)
