#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""
Implements the FallbackForecaster.

The FallbackForecaster works with a list of forecasters and tries to fit them in order. If the active forecaster fails during prediction,
it proceeds to the next. This ensures a robust forecasting mechanism by providing fallback options.
"""

__author__ = ["ninedigits"]
__all__ = ["FallbackForecaster"]

import warnings
from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base import BaseForecaster


class FallbackForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """
    Forecaster that sequentially tries a list of forecasting models.

    Attempts to fit the provided forecasters in the order they are given. If a forecaster fails during fitting or
    prediction, it proceeds to the next one. This class is useful in scenarios where the reliability of individual
    forecasting models may be in question, and a fallback mechanism is desired.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Forecasters to be tried sequentially.
    warn : bool, default=False
        If True, raises warnings when a forecaster fails to fit or predict.
    """

    def __init__(self, forecasters, warn=False):
        super(FallbackForecaster, self).__init__()
        self.forecasters = forecasters
        self.current_forecaster = None
        self.current_name = None
        self.current_forecaster_index = 0
        self.warn = warn
        self._anytagis_then_set("requires-fh-in-fit", True, False, self.forecasters)

    def _fit(self, y, X=None, fh=None):
        """Fit the forecasters in the given order until one succeeds.

        Parameters
        ----------
        y : array-like
            Target time series to which to fit the forecasters.
        X : array-like, optional (default=None)
            Exogenous variables.
        fh : array-like, optional (default=None)
            The forecasting horizon.

        Returns
        -------
        self : an instance of self

        Raises
        ------
        RuntimeError
            If all forecasters fail to fit."""
        self.y = y
        for name, forecaster in self.forecasters[self.current_forecaster_index :]:
            try:
                forecaster.fit(y, X=X, fh=fh)
                self.current_forecaster = forecaster
                self.current_name = name
                return self
            except Exception as e:
                self.current_forecaster_index += 1
                if self.warn:
                    warnings.warn(f"Forecaster {name} failed to fit with error: {e}")

        raise RuntimeError("All forecasters failed to fit.")

    def _predict(self, fh, X=None):
        """Predict using the current forecaster. If it fails, fit and predict with the next forecaster.

        Parameters
        ----------
        fh : array-like
            The forecasting horizon.
        X : array-like, optional (default=None)
            Exogenous variables.

        Returns
        -------
        y_pred : array-like
            The predicted values.

        Raises
        ------
        RuntimeError
            If no forecaster is fitted or all forecasters fail to predict.
        """
        if self.current_forecaster is None:
            raise RuntimeError("No forecaster has been successfully fitted yet.")

        try:
            return self.current_forecaster.predict(fh, X)
        except Exception as e:
            warnings.warn(f"Current forecaster failed at prediction with error: {e}")

            # Remove the failed forecaster
            self.forecasters.pop(0)

            if not self.forecasters:
                raise RuntimeError("No remaining forecasters to attempt prediction.")

            # Fit the next forecaster and retry prediction
            self.current_forecaster = None
            self._fit(self.y, X, fh)
            return self._predict(fh, X)

