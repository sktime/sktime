"""Module for parameter retrieval of impulse responses functions.

This module provides functionality for calculating the impulse responses for 
univariate and multivariate time-series methods, like ARIMA, VAR, VECM, 
VARMAX and other.  

The main class ImpulseResponseFunction provides shall give a unified approach 
for any time series supported by statsmodels.

Supported IRFs according to statsmodels 
(x marks a check done for sktime, given the estimator exists in the library):

- ARIMA
- ARMA
- DynamicFactor
- DynamicFactorMQ
- ExponentialSmoothing
- KalmanFilter
- KalmanSmoother
- MLEModel
- RecursiveLS
- SARIMAX
- SimulationSmoother
- UnobservedComponents
- VAR
- VARMAX
- VECM
"""

__all__ = [
    "ImpulseResponseFunction",
]
from sktime.param_est.impulse._response import ImpulseResponseFunction