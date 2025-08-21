__author__ = ["ltsaprounis"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.estimators._base import _method_logger, _MockEstimatorMixin


class MockUnivariateForecasterLogger(BaseForecaster, _MockEstimatorMixin):
    """Mock univariate forecaster that logs the methods called and their parameters.

    Parameters
    ----------
    prediction_constant : float, optional
        The forecasted value for all steps in the horizon, by default 10

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.utils.estimators import MockUnivariateForecasterLogger
    >>> y = load_airline()
    >>> forecaster = MockUnivariateForecasterLogger()
    >>> forecaster.fit(y)
    MockUnivariateForecasterLogger(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    >>> print(forecaster.log)
    [('_fit', ...), ('_predict', ...)]
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
    }

    def __init__(self, prediction_constant: float = 10):
        self.prediction_constant = prediction_constant
        super().__init__()

    @_method_logger
    def _fit(self, y, X, fh):
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
        return self

    @_method_logger
    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        index = fh.to_absolute_index(self.cutoff)
        return pd.Series(self.prediction_constant, index=index)

    @_method_logger
    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        return self

    @_method_logger
    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        var_names = self._get_varnames()
        var_name = var_names[0]

        fh_index = fh.to_absolute_index(self.cutoff)
        col_index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(columns=col_index, index=fh_index)

        for a in alpha:
            pred_quantiles[(var_name, a)] = pd.Series(
                self.prediction_constant * 2 * a, index=fh_index
            )

        return pred_quantiles


class MockForecaster(BaseForecaster):
    """Mock forecaster, returns data frames filled with a constant.

    Parameters
    ----------
    prediction_constant : float, optional
        The forecasted value for all steps in the horizon, by default 10

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.utils.estimators import MockUnivariateForecasterLogger
    >>> y = load_airline()
    >>> forecaster = MockUnivariateForecasterLogger()
    >>> forecaster.fit(y)
    MockUnivariateForecasterLogger(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    >>> print(forecaster.log)
    [('_fit', ...), ('_predict', ...)]
    """

    _tags = {
        "scitype:y": "both",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
    }

    def __init__(self, prediction_constant: float = 10):
        self.prediction_constant = prediction_constant
        super().__init__()

    def _fit(self, y, X, fh):
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
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        index = fh.to_absolute_index(self.cutoff)
        return pd.DataFrame(
            self.prediction_constant, index=index, columns=self._y.columns
        )

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        return self

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        cols = self._y.columns

        col_index = pd.MultiIndex.from_product([cols, alpha])
        fh_index = fh.to_absolute_index(self.cutoff)
        pred_quantiles = pd.DataFrame(index=fh_index, columns=col_index)

        for col, a in col_index:
            multiple = a if self.prediction_constant > 0 else (1 - a)
            pred_quantiles[col, a] = pd.Series(
                self.prediction_constant * 2 * multiple, index=fh_index
            )

        return pred_quantiles

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [{"prediction_constant": 42}, {"prediction_constant": -4.2}]
