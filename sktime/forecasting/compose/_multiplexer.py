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

    MultiplexForecaster is specified with a (named) list of forecasters
    and a selected_forecaster hyper-parameter, which is one of the forecaster names.
    The MultiplexForecaster then behaves precisely as the forecaster with
    name selected_forecaster, ignoring functionality in the other forecasters.

    When used with ForecastingGridSearchCV, MultiplexForecaster
    provides an ability to tune across multiple estimators, i.e., to perform AutoML,
    by tuning the selected_forecaster hyper-parameter. This combination will then
    select one of the passed forecasters via the tuning algorithm.

    Parameters
    ----------
    forecasters : list
        List of (forecaster names, forecaster objects)
        MultiplexForecaster can switch ("multiplex") between these forecasters.
        These act as blueprints and never change state.
    selected_forecaster: str or None, optional, Default=None.
        If str, must be one of the forecaster names.
        If None, behaves as if the first forecaster in the list is selected.
        Selects the forecaster as which MultiplexForecaster behaves.

    Attributes
    ----------
    forecaster_ : Sktime forecaster
        clone of the selected forecaster used for fitting and forecasting.

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
        selected = self.selected_forecaster
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_forecaster parameter value provided, "
                f" found: {self.selected_forecaster}. Must be one of these"
                f" valid selected_forecaster parameter values: {component_names}."
            )

    def _set_forecaster(self):
        self._check_selected_forecaster()
        # clone the selected forecaster to self.forecaster_
        if self.selected_forecaster is not None:
            for name, forecaster in self._get_estimator_tuples(self.forecasters):
                if self.selected_forecaster == name:
                    self.forecaster_ = clone(forecaster)
        else:
            # if None, simply clone the first forecaster to self.forecaster_
            self.forecaster_ = clone(self._get_estimator_list(self.forecasters)[0])

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
