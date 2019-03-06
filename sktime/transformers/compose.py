"""Meta Transformers module

This module has meta-transformers that is build using the pre-existing
transformers as building blocks.
"""
from .base import BaseTransformer
from ..utils.validation import check_ts_array
from ..utils.transformations import tabularize
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from scipy import sparse
import numpy as np
import pandas as pd


__all__ = ['TSColumnTransformer', 'RowwiseTransformer', 'Tabularizer', 'Tabulariser']

__all__ = ['TSColumnTransformer', 'RowwiseTransformer']


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


class RowwiseTransformer(BaseTransformer):
    """A wrapper for rowwise version of a per-row transformer.

    Takes a per-row transformer and creates a new transformer that could
    repeatedly apply the same transformation to each row in a column of a
    dataframe
    """
    def __init__(self, transformer):
        """
        Parameters
        ----------
        transformer : A transformer that could act on a
            row (one univariate time series array or pd.Series).
            Note that this should be an instance of a transformer class.
        """
        self.transformer = transformer

    def fit(self, X, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        X : 1D array-like, pandas Series, shape (n_samples, 1)
            The training input samples. Shoould not be a DataFrame.
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
        """Calls the fit_transfor() of the per-row transformer repeatedly
        on each row.

        Parameters
        ----------
        X : 1D array-like, pandas Series, shape (n_samples, 1)
            The training input samples. Shoould not be a DataFrame.

        Returns
        -------
        T : 1D array-like, pandas Series, shape (n_samples, ...)
            The transformed data
        """
        # check the validity of input
        X = check_ts_array(X)
        check_is_fitted(self, 'is_fitted_')

        T = X.apply(self.transformer.fit_transform)

        return T


class Tabularizer(BaseTransformer):
    def __init__(self, check_input=True):
        """
        Parameters
        ----------
        transformer : A transformer that could act on a
            row (one univariate time series array or pd.Series).
            Note that this should be an instance of a transformer class.
        """
        self.check_input = check_input

    def fit(self, X, y=None):
        """Empty fit function that does nothing.

        Parameters
        ----------
        X : 1D array-like, pandas Series, shape (n_samples, 1)
            The training input samples. Shoould not be a DataFrame.
        y : None, as it is transformer on X

        Returns
        -------
        self : object
            Returns self.
        """

        # check the validity of input
        # TODO check if for each column, all rows have equal-index series
        if self.check_input:
            X = check_ts_array(X)

        # let the model know that it is fitted
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def transform(self, X):
        """
        Transform nested pandas dataframe into tabular pandas dataframe.
        :param X : pandas dataframe
            Nested dataframe with series or arrays in cells.
        :return : pandas dataframe
            Tabular dataframe with only primitives in cells.
        """

        # check the validity of input
        check_is_fitted(self, 'is_fitted_')

        # TODO check if for each column, all rows have equal-index series
        if self.check_input:
            X = check_ts_array(X)

        Xt = tabularize(X)
        return Xt


Tabulariser = Tabularizer
