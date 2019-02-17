"""
This is a module containing time series transformers
"""
import numpy as np
import pandas as pd
from .utils.validation import check_ts_X_y, check_ts_array, check_is_fitted
from .base import BaseTransformer
from sklearn.compose import ColumnTransformer
from scipy import sparse


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
        # TODO: implement new checks (check_array remoes column names and X is no longer pandas)
        #X = check_ts_array(X)
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


class TSColumnTransformer(ColumnTransformer):
    """Applies transformers to columns of an array or pandas DataFrame.

    Simply takes the column transformer from sklearn and adds capability to handle pandas dataframe
    """

    def __init__(
        self,
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=1,
        transformer_weights=None,
        preserve_dataframe=True,
    ):
        self.preserve_dataframe = preserve_dataframe
        super(TSColumnTransformer, self).__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
        )

    def _hstack(self, Xs):
        """
        Stacks X horizontally.
        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """
        types = set(type(X) for X in Xs)

        if self.sparse_output_:
            return sparse.hstack(Xs).tocsr()
        if self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis="columns")
        return np.hstack(Xs)


    def _validate_output(self, result):
        """
        Ensure that the output of each transformer is 2D. Otherwise
        hstack can raise an error or produce incorrect results.

        Output can also be a pd.Series which is actually a 1D
        """
        names = [name for name, _, _, _ in self._iter(fitted=True,
                                                      replace_strings=True)]
        for Xs, name in zip(result, names):
            if not (getattr(Xs, 'ndim', 0) == 2 or isinstance(Xs, pd.Series)):
                raise ValueError(
                    "The output of the '{0}' transformer should be 2D (scipy " "matrix, array, or pandas DataFrame).".format(name))
