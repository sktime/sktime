"""
This is a module containing time series transformers
"""
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sktime.utils.validation import check_ts_array
from sktime.transformers.base import BaseTransformer


class TSDummyTransformer(BaseTransformer):
    """ An example transformer that makes use of the pandas input.

    Performs the identity transform
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """ A reference implementation of a fitting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        y : None, as it is transformer on X

        Returns
        -------
        self : object
            Returns self.
        """

        # check the validity of input
        X = check_ts_array(X)

        # fitting - this transformer needs no fitting
        pass

        # let the model know that it is fitted
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def transform(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        Returns
        -------
        T : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The transformed data
        """
        # check validity of input
        X = check_ts_array(X)
        check_is_fitted(self, 'is_fitted_')

        T = X

        return T


class TSExampleTransformer(BaseTransformer):
    """ An example transformer that makes use of the pandas input.

    Applies the given tuple of functions to each column
    """

    def __init__(self, funcs=(np.mean, lambda X: np.array(X)[0])):
        self.funcs = funcs

    def fit(self, X, y=None):
        """ A reference implementation of a fitting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        y : None, as it is transformer on X

        Returns
        -------
        self : object
            Returns self.
        """

        # check the validity of input
        X = check_ts_array(X)
        if not X.shape[1] == len(self.funcs):
            raise ValueError("No. of columns and No. of functions supplied to transformer dont match")

        # fitting - this transformer needs no fitting
        pass

        # let the model know that it is fitted
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def transform(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The training input samples.
        Returns
        -------
        T : array-like, pandas DataFrame or Series, shape (n_samples, ...)
            The transformed data
        """
        # TODO: implement new checks (check_array remoes column names and X is no longer pandas)
        #X = check_ts_array(X)
        check_is_fitted(self, 'is_fitted_')

        T = pd.DataFrame([X[col].apply(self.funcs[idx]) for idx, col in enumerate(X.columns)]).T

        return T
