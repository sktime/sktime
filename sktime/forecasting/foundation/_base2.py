"""Shared base class for foundation-model forecasters."""

from sktime.forecasting.base import BaseForecaster


class BaseFoundationForecaster(BaseForecaster):
    """Shared base class for pretrained/foundation forecasting models."""
