# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements the Hyperbolic Sine transformation and its inverse."""

__author__ = ["Ali Parizad"]
__all__ = ["ScaledAsinhTransformer"]


import numpy as np
from scipy import stats

from sktime.transformations.base import BaseTransformer


class ScaledAsinhTransformer(BaseTransformer):
    r"""Hyperbolic sine transformation and its inverse [1]_.

    Known as variance stabilizing transformation,
    Combined with an sktime.forecasting.compose.TransformedTargetForecaster,
    can be usefull in time series that exhibit spikes [1]_, [2]_

    Parameters
    ----------
    mad_normalization_factor : float, default = 1.4826
        The normalization factor used to adjust the median absolute deviation
        (MAD) for asymptotically normal consistency to the standard deviation.
        The default value based on [1]_, [2]_ is 1.4826.

    Attributes
    ----------
    shift_parameter_asinh_ : float
        shift parameter, denoted as "a" in [1]_, the median of sample data.
        It is fitted, based on the data provided in "fit".

    scale_parameter_asinh_ : float
        scale parameter, denoted as "b" in [1]_, the median absolute deviation
        (MAD) around the sample median adjusted by a factor for asymptotically
        normal consistency to the standard deviation (Based on [1]_, [2]_
        b = median_abs_deviation(sample data) :math:`\times` mad_normalization_factor).
        It is fitted, based on the data provided in "fit".

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
    | The Hyperbolic Sine transformation is applied as:
    |   :math:`asinh(\frac{x- a}{b})`
    | The Hyperbolic Sine inverse transformation is applied as:
    |   :math:`b . sinh(x) + a`
    | where "a" is the shift parameter and "b" is the scale parameter [1]_.
    | a = median(sample data)
    | b = median_abs_deviation(sample data) :math:`\times` mad_normalization_factor

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
    >>> from sktime.transformations.series.scaledasinh import ScaledAsinhTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ScaledAsinhTransformer()
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
        "fit_is_empty": False,
        "univariate-only": False,
        "capability:inverse_transform": True,
        "skip-inverse-transform": False,
    }

    def __init__(self, mad_normalization_factor=1.4826):
        super().__init__()
        self.mad_normalization_factor = mad_normalization_factor

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : 2D np.ndarray (n x 1)
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self.shift_parameter_asinh_ = np.median(X)

        self.scale_parameter_asinh_ = (
            stats.median_abs_deviation(X) * self.mad_normalization_factor
        )

        return self

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
        X_transformed = np.arcsinh(
            (X - self.shift_parameter_asinh_) / self.scale_parameter_asinh_
        )

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
        X_inv_transformed = (
            self.scale_parameter_asinh_ * np.sinh(X) + self.shift_parameter_asinh_
        )

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
        params1 = {}
        params2 = {"mad_normalization_factor": 1.6}

        return [params1, params2]
