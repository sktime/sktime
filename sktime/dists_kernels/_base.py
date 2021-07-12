# -*- coding: utf-8 -*-
"""
Abstract base class for pairwise transformers (such as distance/kernel matrix makers)
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator

from sktime.utils.validation.series import check_series


class BasePairwiseTransformer(BaseEstimator):
    """Base pairwise transformer for tabular or series data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
    }

    def __init__(self):
        super().__init__()
        self.X_equals_X2 = False

    def __call__(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X

        alias for transform

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """
        # no input checks or input logic here, these are done in transform
        # this just defines __call__ as an alias for transform
        return self.transform(X=X, X2=X2)

    def transform(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """

        X = check_series(X)

        if X2 is None:
            X2 = X
            self.X_equals_X2 = True
        else:
            X2 = check_series(X2)

            def input_as_numpy(val):
                if isinstance(val, pd.DataFrame):
                    return val.to_numpy(copy=True)
                return val

            temp_X = input_as_numpy(X)
            temp_X2 = input_as_numpy(X2)
            if np.array_equal(temp_X, temp_X2):
                self.X_equals_X2 = True

        return self._transform(X=X, X2=X2)

    def _transform(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

            core logic

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        raise NotImplementedError

    def fit(self, X=None, X2=None):
        # no fitting logic, but in case fit is called or expected
        pass


def _pairwise_panel_x_check(X):
    """
    Method used to check the input and convert
    numpy 3d or numpy list of df to list of dfs

    Parameters
    ----------
    X: List of dfs, Numpy of dfs, 3d numpy
        The value to be checked

    Returns
    -------
    X: List of pd.Dataframe, coerced to this format if one
        of the other formats, otherwise identical to X

    """

    def arr_check(arr):
        for i, Xi in enumerate(arr):
            if not isinstance(Xi, pd.DataFrame):
                raise TypeError(
                    "X must be a list of pd.DataFrame or numpy "
                    "of pd.Dataframe or 3d numpy"
                )
            X[i] = check_series(Xi)

    return_X = []

    if isinstance(X, np.ndarray):
        X_check = np.array(X, copy=True)
        if X_check.ndim == 3:
            for arr in X_check:
                return_X.append(pd.DataFrame(arr))
        else:
            arr_check(X_check)
            return_X = X_check.tolist()
    elif isinstance(X, list):
        arr_check(X)
        return_X = X
    else:
        raise TypeError(
            "X must be a list of pd.DataFrame or numpy of pd.Dataframe or 3d numpy"
        )

    return return_X


class BasePairwiseTransformerPanel(BaseEstimator):
    """Base pairwise transformer for panel data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
    }

    def __init__(self):
        super(BasePairwiseTransformerPanel, self).__init__()
        self.X_equals_X2 = False

    def __call__(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X: list of pd.DataFrame or 2D np.arrays, of length n
        X2: list of pd.DataFrame or 2D np.arrays, of length m, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """
        # no input checks or input logic here, these are done in transform
        # this just defines __call__ as an alias for transform
        return self.transform(X=X, X2=X2)

    def transform(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X: list of pd.DataFrame or 2D np.arrays, of length n
        X2: list of pd.DataFrame or 2D np.arrays, of length m, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]

        Writes to self
        --------------
        X_equals_X2: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """

        X = _pairwise_panel_x_check(X)

        if X2 is None:
            X2 = X
            self.X_equals_X2 = True
        else:
            X2 = _pairwise_panel_x_check(X2)
            if len(X2) == len(X):
                for i in range(len(X2)):
                    if not X[i].equals(X2[i]):
                        break
                self.X_equals_X2 = True

        return self._transform(X=X, X2=X2)

    def _transform(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

            core logic

        Parameters
        ----------
        X: list of pd.DataFrame or 2D np.arrays, of length n
        X2: list of pd.DataFrame or 2D np.arrays, of length m, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        raise NotImplementedError

    def fit(self, X=None, X2=None):
        # no fitting logic, but in case fit is called or expected
        pass
