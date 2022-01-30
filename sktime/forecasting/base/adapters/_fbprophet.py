# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for Facebook prophet to be used in sktime framework."""

__author__ = ["mloning", "aiwalter"]
__all__ = ["_ProphetAdapter"]

import os

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_y_X


class _ProphetAdapter(BaseForecaster):
    """Base class for interfacing prophet and neuralprophet."""

    _tags = {
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def _fit(self, y, X=None, fh=None, **fit_params):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self._instantiate_model()
        self._check_changepoints()
        y, X = check_y_X(y, X, enforce_index_type=pd.DatetimeIndex)

        # We have to bring the data into the required format for fbprophet:
        df = pd.DataFrame({"y": y, "ds": y.index})

        # Add seasonality/seasonalities
        if self.add_seasonality:
            if type(self.add_seasonality) == dict:
                self._forecaster.add_seasonality(**self.add_seasonality)
            elif type(self.add_seasonality) == list:
                for seasonality in self.add_seasonality:
                    self._forecaster.add_seasonality(**seasonality)

        # Add country holidays
        if self.add_country_holidays:
            self._forecaster.add_country_holidays(**self.add_country_holidays)

        # Add regressor (multivariate)
        if X is not None:
            X = X.copy()
            df, X = _merge_X(df, X)
            for col in X.columns:
                self._forecaster.add_regressor(col)

        if self.verbose:
            self._forecaster.fit(df=df, **fit_params)
        else:
            with _suppress_stdout_stderr():
                self._forecaster.fit(df=df, **fit_params)

        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional
            Exogenous data, by default None

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.

        Raises
        ------
        Exception
            Error when merging data
        """
        self._update_X(X, enforce_index_type=pd.DatetimeIndex)

        fh = self.fh.to_absolute(cutoff=self.cutoff).to_pandas()
        if not isinstance(fh, pd.DatetimeIndex):
            raise ValueError("absolute `fh` must be represented as a pd.DatetimeIndex")
        df = pd.DataFrame({"ds": fh}, index=fh)

        # Merge X with df (of created future DatetimeIndex values)
        if X is not None:
            X = X.copy()
            df, X = _merge_X(df, X)

        out = self._forecaster.predict(df)

        out.set_index("ds", inplace=True)
        y_pred = out.loc[:, "yhat"]

        y_pred = pd.DataFrame(y_pred)
        y_pred.reset_index(inplace=True)
        y_pred.index = y_pred["ds"].values
        y_pred.drop("ds", axis=1, inplace=True)
        return y_pred

    def _predict_interval(self, fh, X=None, coverage=0.90):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        fh = fh.to_absolute(cutoff=self.cutoff).to_pandas()
        if not isinstance(fh, pd.DatetimeIndex):
            raise ValueError("absolute `fh` must be represented as a pd.DatetimeIndex")

        # prepare the return DataFrame - empty with correct cols
        var_names = ["Coverage"]
        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx)

        # prepare the DataFrame to pass to prophet
        df = pd.DataFrame({"ds": fh}, index=fh)
        if X is not None:
            X = X.copy()
            df, X = _merge_X(df, X)

        for c in coverage:
            # override parameters in prophet - this is fine since only called in predict
            self._forecaster.interval_width = c
            self._forecaster.uncertainty_samples = self.uncertainty_samples

            # call wrapped prophet, get prediction
            out_prophet = self._forecaster.predict(df)
            # put the index (in ds column) back in the index
            out_prophet.set_index("ds", inplace=True)
            out_prophet.index.name = None
            out_prophet = out_prophet[["yhat_lower", "yhat_upper"]]

            # retrieve lower/upper and write in pred_int return frame
            # instead of writing lower to lower, upper to upper
            #  we take the min/max for lower and upper
            #  because prophet (erroneously?) uses MC indenendent for upper/lower
            #  so if coverage is small, it can happen that upper < lower in prophet
            pred_int[("Coverage", c, "lower")] = out_prophet.min(axis=1)
            pred_int[("Coverage", c, "upper")] = out_prophet.max(axis=1)

        return pred_int

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict

        References
        ----------
        https://facebook.github.io/prophet/docs/additional_topics.html
        """
        self.check_is_fitted()
        fitted_params = {}
        for name in ["k", "m", "sigma_obs"]:
            fitted_params[name] = self._forecaster.params[name][0][0]
        for name in ["delta", "beta"]:
            fitted_params[name] = self._forecaster.params[name][0]
        return fitted_params

    def _check_changepoints(self):
        """Check arguments for changepoints and assign related arguments.

        Returns
        -------
        self
        """
        if self.changepoints is not None:
            self.changepoints = pd.Series(pd.to_datetime(self.changepoints), name="ds")
            self.n_changepoints = len(self.changepoints)
            self.specified_changepoints = True
        else:
            self.specified_changepoints = False
        return self


def _merge_X(df, X):
    """Merge X and df on the DatetimeIndex.

    Parameters
    ----------
    fh : sktime.ForecastingHorizon
    X : pd.DataFrame
        Exog data
    df : pd.DataFrame
        Contains a DatetimeIndex column "ds"

    Returns
    -------
    pd.DataFrame
        DataFrame with containing X and df (with a DatetimeIndex column "ds")

    Raises
    ------
    TypeError
        Error if merging was not possible
    """
    # Merging on the index is unreliable, possibly due to loss of freq information in fh
    X.columns = X.columns.astype(str)
    if "ds" in X.columns and pd.api.types.is_numeric_dtype(X["ds"]):
        longest_column_name = max(X.columns, key=len)
        X.loc[:, str(longest_column_name) + "_"] = X.loc[:, "ds"]
        # raise ValueError("Column name 'ds' is reserved in prophet")
    X.loc[:, "ds"] = X.index
    df = df.merge(X, how="inner", on="ds", copy=True)
    X = X.drop(columns="ds")
    return df, X


class _suppress_stdout_stderr(object):
    """Context manager for doing  a "deep suppression" of stdout and stderr.

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).


    References
    ----------
    https://github.com/facebook/prophet/issues/223
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
