#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _SktimeForecaster

__author__ = ["Kutay Koralturk"]
__all__ = ["Multiplexer"]


class Multiplexer(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    def __init__(
        self,
        components: dict,
        component_fit_params: dict = None,
        select=None,
    ):

        """
        Multiplexer facilitates a framework for performing
        model selection process over different model classes.
        It should be used in conjunction with ForecastingGridSearchCV
        to get full utilization.

        Single use of Multiplexer with components and select
        parameter specified, works just like the selected component.
        It does not provide any further use in that case.

        When used with ForecastingGridSearchCV, multiplexer
        provides an ability to compare different model class
        performances with each other, just like a model tournament.
        When ForecastingGridSearchCV is fitted with a multiplexer,
        returned value for the select argument of best_params_
        attribute of ForecastingGridSearchCV, gives the best
        performing model class among given models provided in components.

        Parameters
        ----------
        components : dict
            A dictionary composed of key-value pairs
            of forecaster names and forecaster objects.
        component_fit_params : dict
            A dictionary composed of key-value pairs
            of forecaster names and fit params to be
            used for each forecaster.
        select: str
            An argument to make a selection among components.
            Multiplexer uses select to choose which component to fit.
            Important for using with ForecastingGridSearchCV as a
            hyperparameter.

        Attributes
        ----------
        _forecaster : Sktime forecaster
            forecaster that multiplexer will currently
            forecast with.
        _forecaster_fit_params: float
            Fit params for the forecaster that is
            going to be used by the forecaster in the
            _forecaster
        """

        self.components = components
        self.component_fit_params = component_fit_params
        self._forecaster = None
        self._forecaster_fit_params = None
        self.select = select
        self._check_components()
        self._check_component_fit_params()

        super(Multiplexer, self).__init__()

    def _check_components(self):
        for key, value in self.components.items():
            if not isinstance(value, _SktimeForecaster):
                raise Exception(
                    "Each component has to be an \
                                sktime forecaster object. \
                                Please check {} value in {} key".format(
                        value, key
                    )
                )

    def _check_component_fit_params(self):
        if self.component_fit_params is None:
            return

        if not (
            all(x in self.components.keys() for x in self.component_fit_params.keys())
        ):
            raise KeyError(
                "If you provide fit_params for models \
                            dictionary key of fit params need to \
                            match the associated component key"
            )

    def _check_select_argument(self):
        if self.select not in self.components.keys():
            raise Exception(
                "Please check the select argument provided  \
                            Valid select parameters: {}".format(
                    self.components.keys()
                )
            )

    def _update_component_fit_params(self):
        self._check_component_fit_params()

        if self.select is None or self.component_fit_params is None:
            return

        if self.select in self.component_fit_params:
            self._forecaster_fit_params = self.component_fit_params[self.select]
        else:
            self._forecaster_fit_params = None

    def _update_forecaster(self):
        self._check_select_argument()
        if self.select is not None:
            self._forecaster = self.components[self.select]

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_y_X(y, X)
        self._set_fh(fh)
        self._check_select_argument()
        self._update_forecaster()
        self._update_component_fit_params()

        if self._forecaster_fit_params:
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
        """Call predict on the forecaster with the best found parameters."""
        self.check_is_fitted()
        self._forecaster.update(y, X, update_params=update_params)
        return self
