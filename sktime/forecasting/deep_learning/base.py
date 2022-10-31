# -*- coding: utf-8 -*-
"""Base Deep Forecaster."""
__author__ = ["AurumnPegasus"]
__all__ = ["BaseDeepForecastor"]

from abc import ABC, abstractmethod

import numpy as np

from sktime.forecasting.base import BaseForecaster


class BaseDeepForecastor(BaseForecaster, ABC):
    """Temp."""

    def __init__(self, batch_size=40):
        super(BaseDeepForecastor).__init__()

        self.batch_size = batch_size
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, **kwargs):
        """Temp."""
        ...

    def _predict(self, fh, X=None):
        """Temp."""
        X = X.transpose((0, 2, 1))
        y_pred = self.model_.predict(X, self.batch_size)
        y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred
