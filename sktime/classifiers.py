"""
This is a module containing time series time_domain_classification
"""
import numpy as np
import pandas as pd
from .utils.validation import check_ts_X_y, check_ts_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from .base import BaseClassifier


class TSDummyClassifier(BaseClassifier):
    """ A dummy classifier to be used as a reference implementation.

    Parameters
    ----------
    strategy : str, default='most'
        A parameter defining the prediction strategy of the dummy classifier
        most: predicts the most frequent class in the dataset
        least: predicts the least frequent class in the dataset
    """
    def __init__(self, strategy='most'):
        self.strategy = strategy

    def fit(self, X, y):
        """ A reference implementation of a fitting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        y : array-like, pandas dataFrame series, shape (n_samples,)
            The target values (class labels in classification)

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_ts_X_y(X, y)
        # convert xpandas or pandas into primitive numpy array (implicit) and get counts
        unique, counts = np.unique(y, return_counts=True)
        # fitting (finding the value of dummy prediciton theta_) the model based on strategy
        if self.strategy == 'most':
            index = np.argmax(counts)
            self.theta_ = unique[index]
        elif self.strategy == 'least':
            index = np.argmin(counts)
            self.theta_ = unique[index]
        else:
            raise ValueError('Unknown Strategy')
        # let the model know that it is fitted
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns the dummy predictions
        """
        X = check_ts_array(X)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64) * self.theta_


class TSExampleClassifier(BaseClassifier):
    """ An example regressor that makes use of the xpandas input.
    """

    def __init__(self, func=np.mean, columns=None, estimator=RandomForestClassifier()):
        self.func = func
        self.columns = columns
        self.estimator = estimator

    def fit(self, X, y):
        """ A reference implementation of a fitting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        y : array-like, pandas dataFrame series, shape (n_samples,)
            The target values (class labels in classification)

        Returns
        -------
        self : object
            Returns self.
        """

        # simple feature extraction
        if self.columns is not None:
            X = pd.DataFrame([X[col].apply(self.func) for col in self.columns]).T

        X, y = check_ts_X_y(X, y)
        # fitting (finding the value of dummy prediction theta_) the model based on strategy

        # fitting
        self.estimator.fit(X, y)

        # let the model know that it is fitted
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns the dummy predictions
        """
        if self.columns is not None:
            X = pd.DataFrame([X[col].apply(self.func) for col in self.columns]).T

        X = check_ts_array(X)

        check_is_fitted(self, 'is_fitted_')

        return self.estimator.predict(X)
