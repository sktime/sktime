"""Interface for the momentfm deep learning time series forecaster."""

# from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster  # , ForecastingHorizon

# if _check_soft_dependencies("momentfm", severity="none"):
#     import momemtfm


__author__ = ["julian-fong"]


class MomentFMForecaster(BaseForecaster):
    """
    Interface for forecasting with the deep learning time series model momentfm.

    MomentFM is a collection of
    """
