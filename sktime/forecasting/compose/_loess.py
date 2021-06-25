#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""STLForecaster Module."""
__all__ = ["STLForecaster"]
__author__ = ["Taiwo Owoseni"]

from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster


class STLForecaster(TransformedTargetForecaster):
    """
    STL - Seasonal and Trend decomposition using Loess.

    STL is a method for decomposing data into three components
    a) Seasonal Component
    b) Trend Component
    c) Residual

    Parameter
    ---------
    forecaster: a forecaster
    degree : int, optional (default=1)
        detrend degree
    sp : int, optional (default=1)
        Seasonal periodicity

    """

    _tags = {"univariate-only": True}
    _required_parameters = ["forecaster"]

    def __init__(self, forecaster=None, degree=1, sp=1):
        self.forecaster = forecaster
        self.sp = sp
        self.degree = degree
        self.detrender = Detrender(
            forecaster=PolynomialTrendForecaster(degree=self.degree)
        )
        self.deseasonalizer = Deseasonalizer(self.sp, model="additive")
        if self.forecaster is None:
            self.forecaster = NaiveForecaster()
        self.steps = [
            ("deseasonalise", self.deseasonalizer),
            ("detrend", self.detrender),
            ("estimator", self.forecaster),
        ]
        super(STLForecaster, self).__init__(self.steps)
