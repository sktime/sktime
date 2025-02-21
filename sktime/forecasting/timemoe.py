"""Implements TimeMOE forecaster."""

__author__ = ["Maple728", "KimMeen", "PranavBhatP"]
# Maple728 and KimMeen for timemoe
__all__ = ["TimeMoEForecaster"]

from sktime.forecasting.base import _BaseGlobalForecaster


class TimeMoEForecaster(_BaseGlobalForecaster):
    """
    Interface for TimeMOE forecaster for zero-shot forecasting.

    TimeMOE is a Mixture-of-Experts model that combines multiple forecasting
    algorithms to make predictions. It is a zero-shot forecasting model that
    can be used to make predictions on unseen time series.
    """
