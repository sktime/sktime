#!/usr/bin/env python3 -u
"""Implements FunctionTransformer, a class to create custom transformers."""

__author__ = ["BoukePostma"]
__all__ = ["FunctionTransformer"]

import numpy as np

from sktime.transformations.base import BaseTransformer


def _identity(X):
    """Return X."""
    return X


class FunctionTransformer(BaseTransformer):
    r"""Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its y (and optionally X) arguments to a
    user-defined function (or callable object) and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda function is used as the ``func``, then the
    resulting transformer will not be pickleable.

    Parameters
    ----------
    func : callable (X: X_type, **kwargs) -> X_type, default=identity (return X)
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.
    inverse_func : callable (X: X_type, **kwargs) -> X_type, default=identity
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
    X_type : str, one of "pd.DataFrame, pd.Series, np.ndarray", or list thereof
        default = ["pd.DataFrame", "pd.Series", "np.ndarray"]
        list of types that func is assumed to allow for X (see signature above)
        if X passed to transform/inverse_transform is not on the list,
            it will be converted to the first list element before passed to funcs

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
        "authors": ["BoukePostma"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series", "np.ndarray"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,
        "handles-missing-data": True,
        "capability:inverse_transform": True,
    }

    def __init__(
        self,
        func=None,
        inverse_func=None,
        *,
        check_inverse=True,
        kw_args=None,
        inv_kw_args=None,
        X_type=None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        self.X_type = X_type
        super().__init__()

        if X_type is not None:
            self.set_tags(X_inner_mtype=X_type)

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

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame or 1D/2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame or 1D/2D np.ndarray, same type as X
            transformed version of X
        """
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X)

        Xt = self._apply_function(X, func=self.func, kw_args=self.kw_args)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        core logic

        Parameters
        ----------
        X : pd.Series or pd.DataFrame or 1D/2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame or 1D/2D np.ndarray, same type as X
            inverse transformed version of X
        """
        Xt = self._apply_function(X, func=self.inverse_func, kw_args=self.inv_kw_args)
        return Xt

    def _apply_function(self, Z, func=None, kw_args=None):
        if func is None:
            func = _identity
        return func(Z, **(kw_args if kw_args else {}))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        # default params, identity transform
        params1 = {}

        # log-transformer, with exp inverse
        params2 = {"func": np.expm1, "inverse_func": np.log1p}

        return [params1, params2]
