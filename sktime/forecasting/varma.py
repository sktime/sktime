# -*- coding: utf-8 -*-
__all__ = ["VARMAX"]
__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX as _VARMAX

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class VARMAX(_StatsModelsAdapter):
    """todo: write docstring.
    """

    _tags = {
        "scitype:y": "multivariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement predict_quantiles?
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        order = (1,0),
        trend = 'c',
        error_cov_type = 'unstructured',
        measurement_error=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
        trend_offset=1,
        start_params=None,
        transformed=True,
        includes_fixed=False,
        cov_type=None,
        cov_kwds=None,
        method='lbfgs',
        maxiter=50,
        full_output=1,
        disp=5,
        callback=None,
        return_params=False,
        optim_score=None,
        optim_complex_step=None,
        optim_hessian=None,
        flags=None,
        low_memory=False,
        dynamic=False,
        information_set='predicted',
        signal_only=False,
        random_state=None,
    ):
        # Model parameters
        self.order = order
        self.trend = trend
        self.error_cov_type = error_cov_type
        self.measurement_error = measurement_error
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.trend_offset = trend_offset
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.optim_score = optim_score
        self.optim_complex_step = optim_complex_step
        self.optim_hessian = optim_hessian
        self.flags = flags
        self.low_memory = low_memory
        self.dynamic = dynamic
        self.information_set = information_set
        self.signal_only = signal_only

        super(VARMAX, self).__init__() # why is this (random_state=random_state) in VAR

    def _fit_forecaster(self, y, X=None): # why error when fit
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """

        self._forecaster = _VARMAX(
            endog=y,
            exog=X,
            order = self.order,
            trend = self.trend,
            error_cov_type = self.error_cov_type,
            measurement_error = self.measurement_error,
            enforce_stationarity = self.enforce_stationarity,
            enforce_invertibility = self.enforce_invertibility,
            trend_offset = self.trend_offset,
        )
        self._fitted_forecaster = self._forecaster.fit(
            start_params = self.start_params,
            transformed = self.transformed,
            includes_fixed = self.includes_fixed,
            cov_type = self.cov_type,
            cov_kwds = self.cov_kwds,
            method = self.method,
            maxiter = self.maxiter,
            full_output = self.full_output,
            disp = self.disp,
            callback = self.callback,
            return_params = self.return_params,
            optim_score = self.optim_score,
            optim_complex_step = self.optim_complex_step,
            optim_hessian = self.optim_hessian,
            flags = self.flags,
            low_memory = self.low_memory,
        )
        return self



    # todo: implement this, mandatory
    def _predict(self, fh, X=None):
        """
        Wrap Statmodel's VARMAX forecast method.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : np.ndarray
            Returns series of predicted values.
        """
        exog_future = X.values if X is not None else None
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]

        return self._fitted_forecaster.predict(start = start,
            end = end,
            dynamic = self.dynamic,
            information_set = self.information_set,
            signal_only = self.signal_only,
            exog = X,
        )


    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params



