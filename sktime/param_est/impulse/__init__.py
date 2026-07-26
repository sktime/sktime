"""Module for parameter retrieval of impulse responses functions.

This module provides functionality for calculating the impulse responses for
univariate and multivariate time-series methods, like ARIMA, VAR, VECM,
VARMAX and other.
"""

__all__ = [
    "ImpulseResponseFunction",
]
from sktime.param_est.impulse._response import ImpulseResponseFunction
