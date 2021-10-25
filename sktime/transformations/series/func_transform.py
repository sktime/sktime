#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Implements FunctionTransformer, a class to create custom transformers."""

__author__ = ["Bouke Postma"]
__all__ = ["FunctionTransformer"]

import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


def _identity(X):
    """Return X."""
    return X


class FunctionTransformer(_SeriesToSeriesTransformer):
    r"""Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its y (and optionally X) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.
    inverse_func : callable, default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.
    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.
    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.
    inv_kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to inverse_func.

    See Also
    --------
    sktime.transformations.series.boxcox.LogTransformer :
        Transformer input data using natural log. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.exponent.ExponentTransformer :
        Transform input data by raising it to an exponent. Can help compress
        variance of series if a fractional exponent is supplied.
    sktime.transformations.series.exponent.SqrtTransformer :
        Transform input data by taking its square root. Can help compress
        variance of input series.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.transformations.series.func_transform import FunctionTransformer
    >>> transformer = FunctionTransformer(np.log1p, np.expm1)
    >>> X = np.array([[0, 1], [2, 3]])
    >>> transformer.fit_transform(X)
    array([[0.        , 0.69314718],
           [1.09861229, 1.38629436]])
    """

    _tags = {
        "handles-missing-data": True,
        "fit-in-transform": False,
    }

    def __init__(
        self,
        func=None,
        inverse_func=None,
        *,
        check_inverse=True,
        kw_args=None,
        inv_kw_args=None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        super(FunctionTransformer, self).__init__()

    def _check_inverse_transform(self, Z):
        """Check that func and inverse_func are each other's inverse."""
        Z_round_trip = self.inverse_func(self.func(Z))
        if not np.allclose(Z_round_trip, Z, equal_nan=True):
            raise UserWarning(
                "The provided functions are not strictly"
                " inverse of each other. If you are sure you"
                " want to proceed regardless, set"
                " 'check_inverse=False'."
            )

    def fit(self, Z, X=None):
        """Fit data.

        Parameters
        ----------
        Z : pd.Series / pd.DataFrame
            Series / DataFrame to fit.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        self
        """
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(Z)

        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Transform data.

        Parameters
        ----------
        Z : pd.Series / pd.DataFrame
            Series / DataFrame to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        Zt : pd.Series / pd.DataFrame
            Transformed data.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        return self._apply_function(Z, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, Z, X=None):
        """Inverse transform data.

        Parameters
        ----------
        Z : pd.Series / pd.DataFrame
            Series / DataFrame to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        Zt : pd.Series / pd.DataFrame
            Inverse transformed data.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        return self._apply_function(Z, func=self.inverse_func, kw_args=self.inv_kw_args)

    def _apply_function(self, Z, func=None, kw_args=None):
        if func is None:
            func = _identity
        return func(Z, **(kw_args if kw_args else {}))
