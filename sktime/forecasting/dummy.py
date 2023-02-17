# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dummy forecasters."""

__author__ = ["fkiraly"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster


class ForecastKnownValues(BaseForecaster):
    """Forecaster that plays back known or prescribed values as forecasts.

    Takes a data set of "known future values" to produces these in the sktime interface.

    Common use cases for this forecaster:

    * as a dummy or naive forecaster with a known baseline expectation
    * as a forecaster with (non-naive) expert forecasts, "known" values as per expert
    * as a counterfactual in benchmarking experiments, "what if we knew the truth"
    * to pass forecast data values in a composite used for postprocessing,
      e.g., in combination with ReconcilerForecaster for an isolated reconciliation step

    When forecasting, uses `pandas.DataFrame.reindex` under the hood to obtain predicted
    values from `y_known`. Paramters other than `y_known` are directly passed
    on to `pandas.DataFrame.reindex`.

    Parameters
    ----------
    y_known : pd.DataFrame
        should contain known values that the forecaster will replay in predict
    method : str or None, optional, default=None
        one of {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}
        method to use for imputing indices at which forecasts are unavailable in y_known
    fill_value : scalar, optional, default=np.NaN
        value to use for any missing values (e.g., if `method` is None)
    limit : int, optional, default=None=infinite
        maximum number of consecutive elements to bfill/ffill if `method=bfill`/`ffill`
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, y_known, method=None, fill_value=None, limit=None):

        self.y_known = y_known
        self.method = method
        self.fill_value = fill_value
        self.limit = limit

        if not isinstance(y_known, pd.DataFrame):
            raise TypeError(
                "y_known parameter of ForecastKnownValues must be pd.DataFrame, "
                f"but found object of type {type(y_known)}"
            )

        super(ForecastKnownValues, self).__init__()

        idx = y_known.index
        if isinstance(idx, pd.MultiIndex):
            if idx.nlevels >= 3:
                mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
            elif idx.levels == 2:
                mtypes = ["pd.DataFrame", "pd-multiindex"]
            self.set_tags(**{"y_inner_mtype": mtypes})
            self.set_tags(**{"X_inner_mtype": mtypes})

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
        # no fitting, we already know the forecast values
        return self

    def _predict(self, fh, X=None):
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
        y_pred : Point predictions
        """
        reindex_params = {"method": self.method, "limit": self.limit}
        if self.fill_value is not None:
            reindex_params["fill_value"] = self.fill_value

        fh_abs = fh.to_absolute(self.cutoff).to_pandas()
        y_pred = self.y_known.reindex(fh_abs, **reindex_params)
        return y_pred

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
        from sktime.utils._testing.series import _make_series

        y = _make_series()
        y2 = y.iloc[3:12]

        params1 = {"y_known": y}
        params2 = {"y_known": y2, "method": "ffill", "limit": 3, "fill_value": 42}

        return [params1, params2]
