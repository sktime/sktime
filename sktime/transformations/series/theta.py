#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova", "Markus Löning"]
# __all__ = ["ThetaLinesTransformer"]

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
    >>> from sktime.transformations.series import ThetaLinesTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ThetaLines([0, 0.25, 0.5, 0.75])
    >>> y_thetas = transformer.transform(y)
    """

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
        thetas: pd.DataFrame
            Transformed series. DataFrame with Theta-lines[1]
            where number of rows = len(Z), number of columns
            = len(self.theta).

        References
        ----------
        [1] E.Spiliotis et al., "Generalizing the Theta method for
        automatic forecasting ", European Journal of Operational
        Research, vol. 284, pp. 550-558, 2020.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)

        if len(self.theta) < 2:
            raise ValueError("`len(theta)` must be >= 2.")

        forecaster = PolynomialTrendForecaster()
        forecaster.fit(z)
        fh = ForecastingHorizon(z.index, is_relative=False)
        trend = forecaster.predict(fh)

        thetas = np.zeros((z.shape[0], len(self.theta)))
        for i, theta in enumerate(self.theta):
            thetas[:, i] = _theta_transform(z, trend, theta)
        return pd.DataFrame(thetas, columns=self.theta)


def _theta_transform(Z, trend, theta):
    # obtain one Theta-line
    z = check_series(Z)

    theta_line = z * theta + (1 - theta) * trend
    return theta_line
