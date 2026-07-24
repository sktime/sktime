"""Implements IBM Granite TSPulse forecaster."""

__author__ = ["Infonioknight"]

__all__ = ["GraniteTSPulseForecaster"]

from sktime.forecasting.base import BaseForecaster


class GraniteTSPulseForecaster(BaseForecaster):
    """Interface to IBM Granite TSPulse forecaster.

    Will populate the details once I figure
    out how to go about this
    """

    _tags = {
        "authors": ["Infonioknight"],
        "maintainers": ["Infonioknight"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        pass

    def _predict(self, fh, X=None):
        pass

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """To be filled once I implement things."""
        return [{}]
