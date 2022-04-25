#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for selecting among different model classes."""

from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster

__author__ = ["kkoralturk", "aiwalter", "fkiraly"]
__all__ = ["MultiplexForecaster"]


class MultiplexForecaster(_DelegatedForecaster, _HeterogenousMetaEstimator):
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
    forecasters : list of sktime forecasters, or
        list of tuples (str, estimator) of sktime forecasters
        MultiplexForecaster can switch ("multiplex") between these forecasters.
        These are "blueprint" forecasters, states do not change when `fit` is called.
    selected_forecaster: str or None, optional, Default=None.
        If str, must be one of the forecaster names.
            If no names are provided, must coincide with auto-generated name strings.
            To inspect auto-generated name strings, call get_params.
        If None, behaves as if the first forecaster in the list is selected.
        Selects the forecaster as which MultiplexForecaster behaves.

    Attributes
    ----------
    forecaster_ : sktime forecaster
        clone of the selected forecaster used for fitting and forecasting.
    forecasters_ : list of (str, forecaster) tuples
        str are identical to those passed, if passed strings are unique
        otherwise unique strings are generated from class name; if not unique,
        the string `_[i]` is appended where `[i]` is count of occurrence up until then
        forecasters in `forecasters_`are reference to forecasters in arg `forecasters`
        i-th forecaster in `forecasters_` is clone of i-th in `forecasters`

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
        "fit_is_empty": False,
    }

    _delegate_name = "forecaster_"

    def __init__(
        self,
        forecasters: list,
        selected_forecaster=None,
    ):
        super(MultiplexForecaster, self).__init__()
        self.selected_forecaster = selected_forecaster

        self.forecasters = forecasters
        self.forecasters_ = self._check_estimators(
            forecasters,
            attr_name="forecasters",
            cls_type=BaseForecaster,
            clone_ests=False,
        )
        self._set_forecaster()
        self.clone_tags(self.forecaster_)
        self.set_tags(**{"fit_is_empty": False})

    def _check_selected_forecaster(self):
        component_names = self._get_estimator_names(self.forecasters_, make_unique=True)
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

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._set_forecaster()
        self.clone_tags(self.forecaster_)
        self.set_tags(**{"fit_is_empty": False})

        super()._fit(y=y, X=X, fh=fh)

        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

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

        params1 = {
            "forecasters": [
                ("Naive_mean", NaiveForecaster(strategy="mean")),
                ("Naive_last", NaiveForecaster(strategy="last")),
                ("Naive_drift", NaiveForecaster(strategy="drift")),
            ],
            "selected_forecaster": "Naive_mean",
        }
        params2 = {
            "forecasters": [
                ("Naive_mean", NaiveForecaster(strategy="mean")),
                ("Naive_last", NaiveForecaster(strategy="last")),
                ("Naive_drift", NaiveForecaster(strategy="drift")),
            ],
        }
        return [params1, params2]
