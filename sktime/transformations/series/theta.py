#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Markus Löning"]
# __all__ = ["ThetaLinesTransformer"]

import numpy as np
import pandas as pd

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster


class ThetaLines(_SeriesToSeriesTransformer):
    """Decompose the original data into two or more ThetaLines."""

    def __init__(self, theta):
        self.theta = theta
        super(ThetaLines, self).__init__()

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
        pd.DataFrame
            Theta lines[1].

        References
        ----------
        [1] E.Spiliotis et al., "Generalizing the Theta method for
        automatic forecasting ", European Journal of Operational
        Research, vol. 284, pp. 550-558, 2020.
        """

        forecaster = PolynomialTrendForecaster()
        forecaster.fit(Z)
        fh = ForecastingHorizon(Z.index, is_relative=False)
        trend = forecaster.predict(fh)

        thetas = np.zeros((Z.shape[0], len(self.theta)))
        for i, theta in enumerate(self.theta):
            thetas[:, i] = _theta_transform(Z, trend, theta)
        return pd.DataFrame(thetas, columns=self.theta)


def _theta_transform(Z, trend, theta):

    theta_line = Z * theta + (1 - theta) * trend
    return theta_line
