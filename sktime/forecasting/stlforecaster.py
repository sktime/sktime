#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""STLForecaster Module."""
__all__ = ["STLForecaster"]
__author__ = ["Taiwo Owoseni"]

from sktime.forecasting.arima import ARIMA
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.compose import TransformedTargetForecaster


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
    steps : list
        Transformers: List of tuples like ("name", transformer)

    Example
    -------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arima import ARIMA
    >>> from sktime.forecasting.stlforecaster import STLForecaster
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> y = load_airline()
    >>> sp = Deseasonalizer()
    >>> dp = Detrender(forecaster=PolynomialTrendForecaster(degree=1))
    >>> estimator = ARIMA(order=(1,1,1))
    >>> forecaster = STLForecaster(estimator, sp, dp)
    >>> forecaster.fit(y)
    STLForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {"univariate-only": True}
    _required_parameters = ["forecaster", "sp", "dp"]

    steps = [
        ("deseasonalise", Deseasonalizer()),
        ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
        ("estimator", ARIMA()),
    ]

    def __init__(self, forecaster=steps[-1][1], sp=steps[0][1], dp=steps[1][1]):
        self.forecaster = forecaster
        self.sp = sp
        self.dp = dp
        super(STLForecaster, self).__init__(STLForecaster.steps)
