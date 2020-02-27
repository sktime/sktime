#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["BaseStatsModelsForecaster"]


from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base import OptionalForecastingHorizonMixin
from sktime.forecasting.base import DEFAULT_ALPHA


class BaseStatsModelsForecaster(OptionalForecastingHorizonMixin, BaseForecaster):

    _fitted_param_names = tuple()

    def __init__(self):
        self._fitted_estimator = None
        super(BaseStatsModelsForecaster, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        """
        Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : array-like, optional (default=[1])
            The forecasters horizon with the steps ahead to to predict.
        X_train : None
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """

        # update observation horizon
        self._set_oh(y_train)
        self._set_fh(fh)

        self._fit_estimator(y_train)

        self._is_fitted = True
        return self

    def _fit_estimator(self, y_train, X_train=None):
        raise NotImplementedError("abstract method")

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Make forecasts.

        Parameters
        ----------
        fh : int or array-like, optional (default=1)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : None
            Exogenous variables are ignored.
        return_pred_int : bool, optional (default=False)
        alpha : int or list, optional (default=0.95)

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        if return_pred_int:
            raise NotImplementedError()
        self._check_is_fitted()
        self._set_fh(fh)

        # Predict fitted model with start and end points relative to start of train series
        fh_abs = self._get_absolute_fh(self.fh)
        start = fh_abs[0]
        end = fh_abs[-1]
        y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon,
        # but only return given time points in pred horizon
        return y_pred.loc[fh_abs]

    def update(self, y_new, X_new=None, update_params=True):
        # input checks
        self._check_is_fitted()

        # update observation horizon
        self._set_oh(y_new)

        if update_params:
            raise NotImplementedError()

        return self

    def get_fitted_params(self):
        self._check_is_fitted()
        return {name: self._fitted_estimator.params.get(name)
                for name in self.get_fitted_param_names()}

    def get_fitted_param_names(self):
        return self._fitted_param_names
