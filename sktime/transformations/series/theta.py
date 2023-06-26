# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements Theta-lines transformation for use with Theta forecasting."""

__author__ = ["GuzalBulatova", "mloning"]
__all__ = ["ThetaLinesTransformer"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.base import BaseTransformer


class ThetaLinesTransformer(BaseTransformer):
    """Decompose the original data into two or more Theta-lines.

    Implementation of decomposition for Theta-method [1]_ as described in [2]_.

    Overview: Input :term:`univariate series <Univariate time series>` of length
    "n" and ThetaLinesTransformer modifies the local curvature of the time series
    using Theta-coefficient values passed through the parameter `theta`.

    Each Theta-coefficient is applied directly to the second differences of the input
    series. The resulting transformed series (Theta-lines) are returned as a
    pd.DataFrame of shape `len(input series) * len(theta)`.

    Parameters
    ----------
    theta : sequence of float, default=(0,2)
        Theta-coefficients to use in transformation.

    Notes
    -----
    Depending on the value of the Theta-coefficient, Theta-lines either augment the
    long-term trend (0 < Theta < 1) or the the short-term behaviour (Theta > 1).

    Special cases:
        - Theta == 0 : deflates input data to linear trend
        - Theta == 1 : returns data unchanged
        - Theta < 0 : transforms time series and mirrors it along the linear trend.

    References
    ----------
    .. [1] V.Assimakopoulos et al., "The theta model: a decomposition approach
       to forecasting", International Journal of Forecasting, vol. 16, pp. 521-530,
       2000.
    .. [2] E.Spiliotis et al., "Generalizing the Theta method for
       automatic forecasting ", European Journal of Operational
       Research, vol. 284, pp. 550-558, 2020.

    Examples
    --------
    >>> from sktime.transformations.series.theta import ThetaLinesTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ThetaLinesTransformer([0, 0.25, 0.5, 0.75])
    >>> y_thetas = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "transform-returns-same-time-index": True,
        "univariate-only": True,
        "fit_is_empty": True,
    }

    def __init__(self, theta=(0, 2)):
        self.theta = theta
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        theta_lines: pd.Series or pd.DataFrame
            Transformed series
            pd.Series, with single Theta-line, if self.theta is float
            pd.DataFrame of shape: [len(X), len(self.theta)], if self.theta is tuple
        """
        theta = _check_theta(self.theta)

        forecaster = PolynomialTrendForecaster()
        forecaster.fit(y=X)
        fh = ForecastingHorizon(X.index, is_relative=False)
        trend = forecaster.predict(fh=fh)

        theta_lines = np.zeros((X.shape[0], len(theta)))
        for i, theta_i in enumerate(theta):
            theta_lines[:, i] = _theta_transform(X, trend, theta_i)
        if isinstance(self.theta, (float, int)):
            return pd.Series(theta_lines.flatten(), index=X.index)
        else:
            return pd.DataFrame(theta_lines, columns=self.theta, index=X.index)


def _theta_transform(Z, trend, theta):
    # obtain one Theta-line
    theta_line = Z * theta + (1 - theta) * trend
    theta_line = theta_line.values.flatten()
    return theta_line


def _check_theta(theta):
    valid_theta_types = (list, int, float, tuple)

    if not isinstance(theta, valid_theta_types):
        raise ValueError(f"invalid input, please use one of {valid_theta_types}")

    if isinstance(theta, (int, float)):
        theta = [theta]

    if isinstance(theta, tuple):
        theta = list(theta)

    return theta
