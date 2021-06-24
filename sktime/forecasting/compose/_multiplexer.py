#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sklearn.base import clone

__author__ = ["Kutay Koralturk"]
__all__ = ["MultiplexForecaster"]


class MultiplexForecaster(_HeterogenousEnsembleForecaster):
    """
    MultiplexForecaster facilitates a framework for performing
    model selection process over different model classes.
    It should be used in conjunction with ForecastingGridSearchCV
    to get full utilization.

    Single use of MultiplexForecaster with forecasters
    and selected_forecaster parameter specified,
    works just like the selected component.
    It does not provide any further use in that case.

    When used with ForecastingGridSearchCV, MultiplexForecaster
    provides an ability to compare different model class
    performances with each other, just like a model tournament.
    When ForecastingGridSearchCV is fitted with a MultiplexForecaster,
    returned value for the selected_forecaster argument of best_params_
    attribute of ForecastingGridSearchCV, gives the best
    performing model class among given models provided in forecasters.

    Parameters
    ----------
    forecasters : list
        List of (forecaster names, forecaster objects)
        MultiplexForecaster switches between these forecasters
        objects when used with ForecastingGridSearchCV to
        find the optimal model
    selected_forecaster: str
        An argument to make a selection among forecasters.
        MultiplexForecaster uses selected_forecaster
        to choose which component to fit.
        Important for using with ForecastingGridSearchCV as a
        hyperparameter.

    Attributes
    ----------
    _forecaster : Sktime forecaster
        forecaster that MultiplexForecaster will currently
        forecast with.

    Example
    ----------
    >>> from sktime.forecasting.all import (
    ...     MultiplexForecaster,
    ...     AutoETS,
    ...     AutoARIMA,
    ...     NaiveForecaster,
    ...     ForecastingGridSearchCV,
    ...     ExpandingWindowSplitter,
    ...     load_airline)
    >>> y = load_airline()
    >>> forecaster = MultiplexForecaster(forecasters=[
    ...     ("ets", AutoETS()),
    ...     ("arima", AutoARIMA(suppress_warnings=True, seasonal=False)),
    ...     ("naive", NaiveForecaster())])
    >>> cv = ExpandingWindowSplitter(
    ...     start_with_window=True,
    ...     step_length=24)
    >>> gscv = ForecastingGridSearchCV(
    ...     cv=cv,
    ...     param_grid={"selected_forecaster":["ets", "arima", "naive"]},
    ...     forecaster=forecaster)
    >>> gscv.fit(y)
    ForecastingGridSearchCV(...)
    >>> gscv.best_forecaster_.selected_forecaster
    'arima'
    """

    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(
        self,
        forecasters: list,
        selected_forecaster=None,
    ):
        super(MultiplexForecaster, self).__init__(forecasters=forecasters, n_jobs=None)
        self.selected_forecaster = selected_forecaster
        self._forecaster = None

    def _check_fit_params(self, fit_params):
        forecaster_fit_params = {}

        if self.selected_forecaster is None or fit_params == {}:
            return forecaster_fit_params

        for component in self.forecasters:
            name, _ = component
            if name not in fit_params.keys():
                raise KeyError(
                    "If you provide fit_params for models "
                    "dictionary key of fit params need to "
                    " match the associated component key."
                )

        if self.selected_forecaster in fit_params.keys():
            forecaster_fit_params = fit_params[self.selected_forecaster]
            return forecaster_fit_params
        else:
            return forecaster_fit_params

    def _check_selected_forecaster(self):
        component_names = [name for name, _ in self.forecasters]
        if self.selected_forecaster not in component_names:
            raise Exception(
                "Please check the selected_forecaster argument provided "
                " Valid selected_forecaster parameters: {}".format(component_names)
            )

    def _set_forecaster(self):
        self._check_selected_forecaster()
        if self.selected_forecaster is not None:
            for name, forecaster in self.forecasters:
                if self.selected_forecaster == name:
                    self._forecaster = clone(forecaster)

    def _fit(self, y, X=None, fh=None, **fit_params):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fit_params : dict
            A dictionary composed of key-value pairs
            of forecaster names and fit params to be
            used for each forecaster.
            Example: {"ARIMA": ..., "ETS": ...}
        Returns
        -------
        self : returns an instance of self.
        """

        self._check_forecasters()
        self._set_forecaster()
        forecaster_fit_params = self._check_fit_params(fit_params=fit_params)
        self._forecaster.fit(y, X=X, fh=fh, **forecaster_fit_params)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=DEFAULT_ALPHA)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        return self._forecaster.predict(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )

    def _update(self, y, X=None, update_params=True):
        """Call predict on the forecaster with the best found parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        self._forecaster.update(y, X, update_params=update_params)
        return self
