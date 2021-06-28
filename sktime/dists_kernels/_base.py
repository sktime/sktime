"""
Abstract base class for pairwise transformers (such as distance/kernel matrix makers)
"""

__author__ = ["fkiraly"]


from sktime.base import BaseEstimator

from sktime.utils.validation.series import check_series


class BaseTrafoPw(BaseEstimator):
    """Base pairwise transformer for tabular or series data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

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
        symmetric: bool = True if X2 was not passed, False if X2 was passed
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
        symmetric: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """

        X = check_series(X)

        if X2 is None:
            X = X2
            self.symmetric = True
        else:
            X2 = check_series(X2)
            self.symmetric = False

        return self.transform(X=X, X2=X2)

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


class BaseTrafoPwPanel(BaseEstimator):
    """Base pairwise transformer for panel data template class.

    The base pairwise transformer specifies the methods and method
    signatures that all pairwise transformers have to implement.

    Specific implementations of these methods is deferred to concrete classes.
    """

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
        symmetric: bool = True if X2 was not passed, False if X2 was passed
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
        symmetric: bool = True if X2 was not passed, False if X2 was passed
            for use to make internal calculations efficient, e.g., in _transform
        """

        if not isinstance(X, list):
            raise TypeError("X must be a list of pd.DataFrame")

        if not isinstance(X2, list):
            raise TypeError("X2 must be a list of pd.DataFrame")

        for i, Xi in enumerate(X):
            X[i] = check_series(Xi)

        if X2 is None:
            X = X2
            self.symmetric = True
        else:
            for i, X2i in enumerate(X2):
                X2[i] = check_series(X2i)
            self.symmetric = False

        return self.transform(X=X, X2=X2)

    def _transform(self, X, X2=None):
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
        """
        raise NotImplementedError
