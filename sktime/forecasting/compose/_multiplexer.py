#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _SktimeForecaster
from sklearn.base import clone

__author__ = ["Kutay Koralturk"]
__all__ = ["MultiplexForecaster"]


class MultiplexForecaster(
    _OptionalForecastingHorizonMixin, _SktimeForecaster, _HeterogenousMetaEstimator
):
    def __init__(
        self,
        forecasters: list,
        selected_forecaster=None,
    ):

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
        _forecaster_fit_params: dict
            Fit params for the forecaster that is
            going to be used by the forecaster in the
            _forecaster
        """

        self.forecasters = forecasters
        self.selected_forecaster = selected_forecaster
        self._check_components()

        super(MultiplexForecaster, self).__init__()

    def _check_components(self):
        if not isinstance(self.forecasters, list):
            raise Exception(
                "Please provide a list " "for forecasters composed of tuples"
            )

        for component in self.forecasters:
            name, forecaster = component
            if not isinstance(forecaster, _SktimeForecaster):
                raise Exception(
                    "Each component has to be an "
                    "sktime forecaster object. "
                    "Please check {} with name:{}".format(forecaster, name)
                )

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

    def _update_forecaster(self):
        self._check_selected_forecaster()
        if self.selected_forecaster is not None:
            for name, forecaster in self.forecasters:
                if self.selected_forecaster == name:
                    self._forecaster = clone(forecaster)

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
        forecaster_fit_params = self._check_fit_params(fit_params=fit_params)
        self._forecaster.fit(y, X=X, fh=fh, **forecaster_fit_params)
        self._is_fitted = True
        return self

    def get_params(self, deep=True):
        """Get parameters for the forecaster.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this forecaster and
            contained subobjects that are forecaster.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("forecasters", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params("forecasters", **kwargs)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        return self._forecaster.predict(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )

    def update(self, y, X=None, update_params=True):
        """Call predict on the forecaster with the best found parameters. """
        self.check_is_fitted()
        self._update_y_X(y, X)
        self._forecaster.update(y, X, update_params=update_params)
        return self
