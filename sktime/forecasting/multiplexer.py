#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _SktimeForecaster
import copy

__author__ = ["Kutay Koralturk"]
__all__ = ["MultiplexerForecaster"]


class MultiplexerForecaster(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    def __init__(
        self,
        component_estimators: list,
        selected_estimator=None,
    ):

        """
        MultiplexerForecaster facilitates a framework for performing
        model selection process over different model classes.
        It should be used in conjunction with ForecastingGridSearchCV
        to get full utilization.

        Single use of MultiplexerForecaster with component_estimators
        and selected_estimator parameter specified,
        works just like the selected component.
        It does not provide any further use in that case.

        When used with ForecastingGridSearchCV, MultiplexerForecaster
        provides an ability to compare different model class
        performances with each other, just like a model tournament.
        When ForecastingGridSearchCV is fitted with a MultiplexerForecaster,
        returned value for the selected_estimator argument of best_params_
        attribute of ForecastingGridSearchCV, gives the best
        performing model class among given models provided in component_estimators.

        Parameters
        ----------
        component_estimators : list
            List of (forecaster names, forecaster objects)
            MultiplexerForecaster switches between these forecasters
            objects when used with ForecastingGridSearchCV to
            find the optimal model
        selected_estimator: str
            An argument to make a selection among component_estimators.
            MultiplexerForecaster uses selected_estimator
            to choose which component to fit.
            Important for using with ForecastingGridSearchCV as a
            hyperparameter.

        Attributes
        ----------
        _forecaster : Sktime forecaster
            forecaster that MultiplexerForecaster will currently
            forecast with.
        _forecaster_fit_params: dict
            Fit params for the forecaster that is
            going to be used by the forecaster in the
            _forecaster
        """

        self.component_estimators = component_estimators
        self.selected_estimator = selected_estimator
        self._check_components()
        self._forecaster_fit_params = None

        super(MultiplexerForecaster, self).__init__()

    def _check_components(self):
        if not isinstance(self.component_estimators, list):
            raise Exception(
                "Please provide a list " "for component_estimators composed of tuples"
            )

        for component in self.component_estimators:
            name, estimator = component
            if not isinstance(estimator, _SktimeForecaster):
                raise Exception(
                    "Each component has to be an "
                    "sktime forecaster object. "
                    "Please check {} with name:{}".format(estimator, name)
                )

    def _check_fit_params(self, fit_params):
        if fit_params is None or fit_params == {}:
            return

        for component in self.component_estimators:
            name, _ = component

            if name not in fit_params.keys():
                raise KeyError(
                    "If you provide fit_params for models "
                    "dictionary key of fit params need to "
                    " match the associated component key."
                )

    def _check_selected_estimator(self):
        component_names = [name for name, _ in self.component_estimators]
        if self.selected_estimator not in component_names:
            raise Exception(
                "Please check the selected_estimator argument provided "
                " Valid selected_estimator parameters: {}".format(component_names)
            )

    def _update_forecaster_fit_params(self, fit_params):
        self._check_fit_params(fit_params)

        if self.selected_estimator is None or fit_params is None:
            return

        if self.selected_estimator in fit_params.keys():
            self._forecaster_fit_params = self.component_fit_params[
                self.selected_estimator
            ]
        else:
            self._forecaster_fit_params = None

    def _update_forecaster(self):
        self._check_selected_estimator()
        if self.selected_estimator is not None:
            for name, estimator in self.component_estimators:
                if self.selected_estimator == name:
                    self._forecaster = copy.deepcopy(estimator)

    def fit(self, y, X=None, fh=None, **fit_params):
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
        self._set_y_X(y, X)
        self._set_fh(fh)
        self._update_forecaster()
        self._update_forecaster_fit_params(fit_params=fit_params)

        if fit_params:
            self._forecaster.fit(y, X=X, fh=fh, **self._forecaster_fit_params)
        else:
            self._forecaster.fit(y, X=X, fh=fh)

        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        return self._forecaster.predict(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )

    def update(self, y, X=None, update_params=False):
        """Call predict on the forecaster with the best found parameters. """
        self.check_is_fitted()
        self._update_y_X(y, X)
        self._forecaster.update(y, X, update_params=update_params)
        return self
