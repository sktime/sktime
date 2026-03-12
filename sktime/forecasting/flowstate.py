"""Implements Granite FlowState Time Series Forecaster. Ref Issue: #9632."""

__author__ = ["FlyingDragon112"]
__all__ = ["GraniteFlowStateForecaster"]

from sktime.forecasting.base import BaseForecaster


class GraniteFlowStateForecaster(BaseForecaster):
    """
    Granite FlowState Time Series Forecaster.

    Status: Researching and Implementing
    """

    _tags = {
        "authors": ["FlyingDragon112"],
        "maintainers": ["FlyingDragon112"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        pass

    def _predict(self, y, X=None, fh=None):
        pass

    def _update(self, y_new, X_new=None):
        pass

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """To be filled once I implement things."""
        return [{}]
