# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements the Hyperbolic Sine transformation and its inverse."""

__author__ = ["Ali Parizad"]
__all__ = ["ScaledAsinhTransformer"]

from copy import deepcopy

import numpy as np

from sktime.transformations.base import BaseTransformer


class ScaledAsinhTransformer(BaseTransformer):
    r"""Hyperbolic sine transformation and its inverse [1]_.

    Known as variance stabilizing transformation [2]_,
    Combined with an sktime.forecasting.compose.TransformedTargetForecaster,
    can be usefull in time series that exhibit spikes [1]_, [2]_

    Parameters
    ----------
    shift_parameter_asinh : float, optional, default=None
        shift parameter, denoted as "a" in [1]_, the median of sample data.
    scale_parameter_asinh : float, optional, default=None
        scale parameter, denoted as "b" in [1]_, the median absolute deviation
        (MAD) around the sample median adjusted by a factor for asymptotically
        normal consistency to the standard deviation (Based on [2]_, b= 1.4826)

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
    | The Hyperbolic Sine transformation is applied if both shift_parameter_asinh and
    | scale_parameter_asinh are not None:
    |   :math:`X_transform  = asinh(\frac{X - a}{b})`
    |   :math:`X_inverse_transform  = b . sinh(X) + a`
    | where a is the shift parameter and b is the scale parameter [1]_.

    References
    ----------
    .. [1] Ziel F, Weron R. Day-ahead electricity price forecasting with
        high-dimensional structures: Univariate vs. multivariate modeling
        frameworks. Energy Economics. 2018 Feb 1;70:396-420.
    .. [2] Uniejewski, B., Weron, R., Ziel, F., 2017. Variance stabilizing
        transformations for electricity spot price forecasting.
        IEEE Transactions on Power Systems, DOI: 10.1109/TPWRS.2017.2734563

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.transformations.series.scaledasinh import ScaledAsinhTransformer
    >>> from sktime.datasets import load_airline
    >>> from scipy import stats
    >>> y =  load_airline()
    >>> shift_parameter_asinh = np.median(y)
    >>> scale_parameter_asinh = stats.median_abs_deviation(y) * 1.4826
    >>> transformer = ScaledAsinhTransformer(shift_parameter_asinh,
    ... scale_parameter_asinh)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
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

    def __init__(self, shift_parameter_asinh=None, scale_parameter_asinh=None):
        self.shift_parameter_asinh = shift_parameter_asinh
        self.scale_parameter_asinh = scale_parameter_asinh
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
        if self.scale_parameter_asinh and self.shift_parameter_asinh:
            X_transformed = np.arcsinh(
                (X - self.shift_parameter_asinh) / self.scale_parameter_asinh
            )

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
        if self.scale_parameter_asinh and self.shift_parameter_asinh:
            X_inv_transformed = (
                self.scale_parameter_asinh * np.sinh(X) + self.shift_parameter_asinh
            )

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
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        test_params = [
            {"shift_parameter_asinh": None, "scale_parameter_asinh": None},
            {"shift_parameter_asinh": 5.4, "scale_parameter_asinh": 3.7},
        ]
        return test_params
