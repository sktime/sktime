"""
Base estimator for transformers
"""
import abc
from sklearn.base import BaseEstimator


class BaseTransformer(BaseEstimator, metaclass=abc.ABCMeta):
    """
    Base class for transformers, for identification.
    """
    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Function to fit transformer
        As fit_transform calls it, its implementation is made mandatory
        """

    @abc.abstractmethod
    def transform(self, X, y=None):
        """
        Function to perform actual transformation
        As fit_transform calls it, its implementation is made mandatory
        """

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
