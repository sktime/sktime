# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements the scaled logit transformation."""

__author__ = ["ltsaprounis"]
__all__ = ["ScaledLogitTransformer"]

from copy import deepcopy

import numpy as np

from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn


class ScaledLogitTransformer(BaseTransformer):
    r"""Scaled logit transform or Log transform.

    If both lower_bound and upper_bound are not None, a scaled logit transform is
    applied to the data. Otherwise, the transform applied is a log transform variation
    that ensures the resulting values from the inverse transform are bounded
    accordingly. The transform is applied to all scalar elements of the input array
    individually.

    Combined with an sktime.forecasting.compose.TransformedTargetForecaster, it ensures
    that the forecast stays between the specified bounds (lower_bound, upper_bound).

    Default is lower_bound = upper_bound = None, i.e., the identity transform.

    The logarithm transform is obtained for lower_bound = 0, upper_bound = None.

    Parameters
    ----------
    lower_bound : float, optional, default=None
        lower bound of inverse transform function
    upper_bound : float, optional, default=None
        upper bound of inverse transform function

    See Also
    --------
    sktime.transformations.series.boxcox.LogTransformer :
        Transformer input data using natural log. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.boxcox.BoxCoxTransformer :
        Applies Box-Cox power transformation. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.exponent.ExponentTransformer :
        Transform input data by raising it to an exponent. Can help compress
        variance of series if a fractional exponent is supplied.
    sktime.transformations.series.exponent.SqrtTransformer :
        Transform input data by taking its square root. Can help compress
        variance of input series.

    Notes
    -----
    | The scaled logit transform is applied if both upper_bound and lower_bound are
    | not None:
    |   :math:`log(\frac{x - a}{b - x})`, where a is the lower and b is the upper bound.

    | If upper_bound is None and lower_bound is not None the transform applied is
    | a log transform of the form:
    |   :math:`log(x - a)`

    | If lower_bound is None and upper_bound is not None the transform applied is
    | a log transform of the form:
    |   :math:`- log(b - x)`

    References
    ----------
    .. [1] Hyndsight - Forecasting within limits:
        https://robjhyndman.com/hyndsight/forecasting-within-limits/
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3.
        Accessed on January 24th 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.scaledlogit import ScaledLogitTransformer
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> y = load_airline()
    >>> fcaster = TransformedTargetForecaster([
    ...     ("scaled_logit", ScaledLogitTransformer(0, 650)),
    ...     ("poly", PolynomialTrendForecaster(degree=2))
    ... ])
    >>> fcaster.fit(y)
    TransformedTargetForecaster(...)
    >>> y_pred = fcaster.predict(fh = np.arange(32))
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ltsaprounis"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "transform-returns-same-time-index": True,
        "fit_is_empty": True,
        "univariate-only": False,
        "capability:inverse_transform": True,
        "skip-inverse-transform": False,
    }

    def __init__(self, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Ignored argument for interface compatibility

        Returns
        -------
        transformed version of X
        """
        if self.upper_bound is not None and np.any(X >= self.upper_bound):
            warn(
                "X in ScaledLogitTransformer should not have values "
                "greater than upper_bound",
                RuntimeWarning,
                obj=self,
            )

        if self.lower_bound is not None and np.any(X <= self.lower_bound):
            warn(
                "X in ScaledLogitTransformer should not have values "
                "lower than lower_bound",
                RuntimeWarning,
                obj=self,
            )

        if self.upper_bound and self.lower_bound:
            X_transformed = np.log((X - self.lower_bound) / (self.upper_bound - X))
        elif self.upper_bound is not None:
            X_transformed = -np.log(self.upper_bound - X)
        elif self.lower_bound is not None:
            X_transformed = np.log(X - self.lower_bound)
        else:
            X_transformed = deepcopy(X)

        return X_transformed

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Ignored argument for interface compatibility

        Returns
        -------
        inverse transformed version of X
        """
        if self.upper_bound and self.lower_bound:
            X_inv_transformed = (self.upper_bound * np.exp(X) + self.lower_bound) / (
                np.exp(X) + 1
            )
        elif self.upper_bound is not None:
            X_inv_transformed = self.upper_bound - np.exp(-X)
        elif self.lower_bound is not None:
            X_inv_transformed = np.exp(X) + self.lower_bound
        else:
            X_inv_transformed = deepcopy(X)

        return X_inv_transformed

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
        test_params = [
            {"lower_bound": None, "upper_bound": None},
            {"lower_bound": -(10**6), "upper_bound": None},
            {"lower_bound": None, "upper_bound": 10**6},
            {"lower_bound": -(10**6), "upper_bound": 10**6},
        ]
        return test_params
