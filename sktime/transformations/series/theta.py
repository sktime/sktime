#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova", "Markus Löning"]
__all__ = ["ThetaLinesTransformer"]

import numpy as np
import pandas as pd

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.utils.validation.series import check_series


class ThetaLinesTransformer(_SeriesToSeriesTransformer):
    """Decompose the original data into two or more Theta-lines.

    Example
    -------
    >>> from sktime.transformations.series.theta import ThetaLinesTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ThetaLinesTransformer([0, 0.25, 0.5, 0.75])
    >>> y_thetas = transformer.fit_transform(y)

    References
    ----------
    [1] E.Spiliotis et al., "Generalizing the Theta method for
    automatic forecasting ", European Journal of Operational
    Research, vol. 284, pp. 550-558, 2020.
    """

    _tags = {
        "transform-returns-same-time-index": True,
        "univariate-only": True,
        "fit-in-transform": True,
    }

    def __init__(self, theta=(0, 2)):
        self.theta = theta
        super(ThetaLinesTransformer, self).__init__()

    def transform(self, Z, X=None):
        """Transform data.

        Parameters
        ----------
        Z : pd.Series
            Series to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        theta_lines: ndarray or pd.DataFrame
            Transformed series: single Theta-line or a pd.DataFrame of
            shape: len(Z)*len(self.theta).
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        theta = _check_theta(self.theta)

        forecaster = PolynomialTrendForecaster()
        forecaster.fit(z)
        fh = ForecastingHorizon(z.index, is_relative=False)
        trend = forecaster.predict(fh)

        theta_lines = np.zeros((z.shape[0], len(theta)))
        for i, theta in enumerate(theta):
            theta_lines[:, i] = _theta_transform(z, trend, theta)
        if isinstance(self.theta, (float, int)):
            return pd.Series(theta_lines.flatten(), index=z.index)
        else:
            return pd.DataFrame(theta_lines, columns=self.theta, index=z.index)


def _theta_transform(Z, trend, theta):
    # obtain one Theta-line
    theta_line = Z * theta + (1 - theta) * trend
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
