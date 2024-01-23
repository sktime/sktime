#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""
Implements the FallbackForecaster.

The FallbackForecaster works with a list of forecasters and tries to fit them in order.
If the active forecaster fails during prediction, it proceeds to the next. This ensures
a robust forecasting mechanism by providing fallback options.
"""

__author__ = ["ninedigits"]
__all__ = ["FallbackForecaster"]

import warnings

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base import BaseForecaster


class FallbackForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """Forecaster that sequentially tries a list of forecasting models.

    Attempts to fit the provided forecasters in the order they are given. If a
    forecaster fails during fitting or prediction, it proceeds to the next one. This
    class is useful in scenarios where the reliability of individual forecasting models
    may be in question, and a fallback mechanism is desired.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Forecasters to be tried sequentially.
    warn : bool, default=False
        If True, raises warnings when a forecaster fails to fit or predict.

    Examples
    --------
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import FallbackForecaster
    >>> from sktime.forecasting.compose import EnsembleForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [
    ...     (
    ...         "ensemble",
    ...         EnsembleForecaster(
    ...             [
    ...                 ("trend", PolynomialTrendForecaster()),
    ...                 ("polynomial", NaiveForecaster())
    ...             ]
    ...         )
    ...     ),
    ...     ("naive", NaiveForecaster())
    ... ]
    >>> forecaster = FallbackForecaster(forecasters=forecasters)
    >>> forecaster.fit(y=y, fh=[1, 2, 3])
    FallbackForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    def __init__(self, forecasters, warn=False):
        super().__init__()
        self.forecasters = tuple(forecasters)
        self.current_forecaster = None
        self.current_name = None
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
            If all forecasters fail to fit.
        """
        self.first_nonfailing_forecaster_index_ = 0
        self.exceptions_raised_ = dict()
        return self._try_fit_forecasters()

    def _try_fit_forecasters(self):
        """
        Attempt to fit the forecasters in sequence until one succeeds.

        This method iterates over the forecasters starting from the index
        `first_nonfailing_forecaster_index_`. For each forecaster, it tries to fit it
        with the current data. If the fit method of a forecaster raises an exception,
        it records the exception and proceeds to the next forecaster. If a forecaster
        fits successfully, it updates the current forecaster and its name.

        Returns
        -------
        self : an instance of self

        Raises
        ------
        RuntimeError
            If all forecasters fail to fit.
        """
        while True:
            if self.first_nonfailing_forecaster_index_ >= len(self.forecasters):
                raise RuntimeError("No remaining forecasters to attempt prediction.")
            name, forecaster = self.forecasters[self.first_nonfailing_forecaster_index_]
            try:
                self.current_name = name
                self.current_forecaster = forecaster.clone()
                self.current_forecaster.fit(self._y, X=self._X, fh=self._fh)
                return self
            except Exception as e:
                self.exceptions_raised_[self.first_nonfailing_forecaster_index_] = {
                    "failed_at_step": "fit",
                    "exception": e,
                    "forecaster_name": name,
                }
                self.first_nonfailing_forecaster_index_ += 1
                if self.warn:
                    warnings.warn(
                        f"Forecaster {name} failed to fit with error: {e}", stacklevel=2
                    )

    def _predict(self, fh, X=None):
        """Predict using the current forecaster.

        If predict fails, fit and predict with the next forecaster.

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
            self.exceptions_raised_[self.first_nonfailing_forecaster_index_] = {
                "failed_at_step": "fit",
                "exception": e,
                "forecaster_name": self.current_name,
            }
            if self.warn:
                warnings.warn(
                    f"Current forecaster failed at prediction with error: {e}",
                    stacklevel=2,
                )
            self.first_nonfailing_forecaster_index_ += 1

            # Fit the next forecaster and retry prediction
            self.current_forecaster = None
            self._try_fit_forecasters()
            return self.predict(fh, X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.compose._reduce import DirectReductionForecaster
        from sktime.forecasting.naive import NaiveForecaster

        # univariate case
        FORECASTER = NaiveForecaster()
        params = [{"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}]

        # test multivariate case, i.e., ensembling multiple variables at same time
        FORECASTER = DirectReductionForecaster.create_test_instance()
        params = params + [{"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}]

        return params
