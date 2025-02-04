"""Scikit-learn transformer to compute fractional differentiation."""

from typing import TypeVar

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import (
    BaseEstimator,  # type: ignore
    TransformerMixin,  # type: ignore
)
from sklearn.utils.validation import check_is_fitted

from sktime.libs.fracdiff.fdiff import fdiff, fdiff_coef

T = TypeVar("T", bound="Fracdiff")


sklearn_ge_16 = _check_soft_dependencies("scikit-learn>=1.6.0", severity="none")

if sklearn_ge_16:
    from sklearn.utils.validation import validate_data as _skl_checker
else:
    from sklearn.utils.validation import check_array as _skl_checker


def _sklearn_check_input(*args, **kwargs):
    """Downwards compatibility switch for new input checks from scikit-learn 1.6 on.

    Parameters
    ----------
    method : str, default None
        Method from which the check is called, e.g., "fit" or "transform".
    """
    method = kwargs.pop("method", None)

    if sklearn_ge_16 and method != "fit":
        kwargs.update({"reset": False})
    if sklearn_ge_16:
        # from sklearn 1.6.0, the estimator arg is called _estimator
        est = kwargs.pop("estimator", None)
        args = (est,) + args
    else:
        X = kwargs.pop("X", None)
        args = (X,) + args

    return _skl_checker(*args, **kwargs)


class Fracdiff(TransformerMixin, BaseEstimator):
    r"""A scikit-learn transformer to compute fractional differentiation.

    Parameters
    ----------
    d : float, default 1.0
        The order of differentiation.
    window : int > 0 or None, default 10
        Number of observations to compute each element in the output.
    mode : {"same", "valid"}, default "same"
        See :func:`fracdiff.fdiff` for details.
    window_policy : {"fixed"}, default "fixed"
        "fixed" (default) :
            Fixed window method.
            Every term in the output is evaluated using `window` observations.
            In other words, a fracdiff operator, which is a polynominal of a backshift
            operator, is truncated up to the `window`-th term.
            The beginning window\_ - 1 elements in output are filled with `numpy.nan`.
        "expanding" (not available) :
            Expanding window method.
            Every term in fracdiff time-series is evaluated using at least `window`
            observations.
            The beginning `window - 1` elements in output are filled with `numpy.nan`.

    Attributes
    ----------
    coef_ : numpy.array, shape (window,)
        Sequence of coefficients in the fracdiff operator.

    Examples
    --------
    >>> from fracdiff.sklearn import Fracdiff
    >>> X = numpy.arange(10).reshape(5, 2)
    >>> fracdiff = Fracdiff(0.5, window=3)
    >>> fracdiff.fit_transform(X)
    array([[0.   , 1.   ],
           [2.   , 2.5  ],
           [3.   , 3.375],
           [3.75 , 4.125],
           [4.5  , 4.875]])
    >>> fracdiff.coef_
    array([ 1.   , -0.5  , -0.125])

    >>> fracdiff = Fracdiff(0.5, window=3, mode="valid")
    >>> fracdiff.fit_transform(X)
    array([[3.   , 3.375],
           [3.75 , 4.125],
           [4.5  , 4.875]])

    >>> X = numpy.array([1, 0, 0, 0]).reshape(-1, 1)
    >>> fracdiff = Fracdiff(0.5, window=4)
    >>> fracdiff.fit_transform(X)
    array([[ 1.    ],
           [-0.5   ],
           [-0.125 ],
           [-0.0625]])
    """

    def __init__(
        self,
        d: float = 1.0,
        window: int = 10,
        mode: str = "same",
        window_policy: str = "fixed",
    ) -> None:
        self.d = d
        self.window = window
        self.mode = mode
        self.window_policy = window_policy

    def __repr__(self) -> str:
        """Repr string of the object.

        Examples
        --------
        >>> Fracdiff(0.5)
        Fracdiff(d=0.5, window=10, mode=same, window_policy=fixed)
        """
        name = self.__class__.__name__
        attrs = ["d", "window", "mode", "window_policy"]
        params = ", ".join(f"{attr}={getattr(self, attr)}" for attr in attrs)
        return f"{name}({params})"

    def fit(self: T, X: np.ndarray, y: None = None) -> T:
        """
        Fit the model with `X`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Time-series to perform fractional differentiation.
            Here `n_samples` is the number of samples and `n_features` is the number of
            features.
        y : array_like, optional
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = _sklearn_check_input(X=X, estimator=self, method="fit")
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1]
        self.coef_ = fdiff_coef(self.d, self.window)
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Return the fractional differentiation of `X`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_series)
            Time-series to perform fractional differentiation.
            Raises ValueError if `n_samples < self.window`.
        y : array_like, optional
            Ignored.

        Returns
        -------
        fdiff : ``numpy.array``, shape (n_samples, n_series)
            The fractional differentiation of `X`.
        """
        check_is_fitted(self, ["coef_"])
        X = _sklearn_check_input(X=X, estimator=self, method="transform")

        # Check that the number of features in transform matches fit
        if hasattr(X, "shape") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Error in Fracdiff transformer: "
                f"Number of features in transform ({X.shape[1]}) does not match "
                f"number of features in fit ({self.n_features_in_})."
            )

        Xt = fdiff(X, n=self.d, axis=0, window=self.window, mode=self.mode)
        return Xt
