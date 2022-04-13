#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for selecting among different model classes."""

from sklearn.base import clone

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster

__author__ = ["kkoralturk", "aiwalter", "fkiraly"]
__all__ = ["MultiplexForecaster"]


class MultiplexForecaster(_HeterogenousEnsembleForecaster, _DelegatedForecaster):
    """MultiplexForecaster for selecting among different models.

    MultiplexForecaster facilitates a framework for performing
    model selection process over different model classes.
    It should be used in conjunction with ForecastingGridSearchCV
    to get full utilization. It can be used with univariate and
    multivariate forecasters.

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
    forecaster_ : Sktime forecaster
        forecaster that MultiplexForecaster will currently
        forecast with.

    Examples
    --------
    >>> from sktime.forecasting.all import (
    ...     MultiplexForecaster,
    ...     AutoETS,
    ...     AutoARIMA,
    ...     NaiveForecaster,
    ...     ForecastingGridSearchCV,
    ...     ExpandingWindowSplitter,
    ...     load_shampoo_sales)
    >>> y = load_shampoo_sales()
    >>> forecaster = MultiplexForecaster(forecasters=[
    ...     ("ets", AutoETS()),
    ...     ("arima", AutoARIMA(suppress_warnings=True, seasonal=False)),
    ...     ("naive", NaiveForecaster())])
    >>> cv = ExpandingWindowSplitter(
    ...     start_with_window=True,
    ...     step_length=12)
    >>> gscv = ForecastingGridSearchCV(
    ...     cv=cv,
    ...     param_grid={"selected_forecaster":["ets", "arima", "naive"]},
    ...     forecaster=forecaster)
    >>> gscv.fit(y)
    ForecastingGridSearchCV(...)
    """

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "scitype:y": "both",
        "y_inner_mtype": ["pd.DataFrame", "pd.Series"],
    }

    _delegate_name = "forecaster_"

    def __init__(
        self,
        forecasters: list,
        selected_forecaster=None,
    ):
        super(MultiplexForecaster, self).__init__(forecasters=forecasters, n_jobs=None)
        self.selected_forecaster = selected_forecaster

        self._check_forecasters()
        self._set_forecaster()

        self.clone_tags(selected_forecaster)

    def _check_selected_forecaster(self):
        component_names = self._get_estimator_names(self.forecasters, make_unique=True)
        if self.selected_forecaster not in component_names:
            raise Exception(
                "Please check the selected_forecaster argument provided "
                " Valid selected_forecaster parameters: {}".format(component_names)
            )

    def _set_forecaster(self):
        self._check_selected_forecaster()
        if self.selected_forecaster is not None:
            for name, forecaster in self._get_estimator_tuples(self.forecasters):
                if self.selected_forecaster == name:
                    self.forecaster_ = clone(forecaster)

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
        from sktime.forecasting.naive import NaiveForecaster

        params = {
            "forecasters": [
                ("Naive_mean", NaiveForecaster(strategy="mean")),
                ("Naive_last", NaiveForecaster(strategy="last")),
                ("Naive_drift", NaiveForecaster(strategy="drift")),
            ],
            "selected_forecaster": "Naive_mean",
        }
        return params
