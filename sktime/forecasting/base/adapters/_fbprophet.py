#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "Martin Walter"]
__all__ = ["_ProphetAdapter"]

import os

import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base import BaseForecaster
from contextlib import contextmanager


class _ProphetAdapter(BaseForecaster):
    """Base class for interfacing fbprophet and neuralprophet"""

    _tags = {
        "univariate-only": False,
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
        self._set_y_X(y, X, enforce_index_type=pd.DatetimeIndex)

        # We have to bring the data into the required format for fbprophet:
        df = pd.DataFrame({"y": y, "ds": y.index})

        # Add seasonality
        if self.add_seasonality:
            self._forecaster.add_seasonality(**self.add_seasonality)

        # Add country holidays
        if self.add_country_holidays:
            self._forecaster.add_country_holidays(**self.add_country_holidays)

        # Add regressor (multivariate)
        if X is not None:
            df, X = _merge_X(df, X)
            for col in X.columns:
                self._forecaster.add_regressor(col)

        if self.verbose:
            self._forecaster.fit(df=df, **fit_params)
        else:
            with _suppress_stdout_stderr():
                self._forecaster.fit(df=df, **fit_params)

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Predict

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional
            Exogenous data, by default None
        return_pred_int : bool, optional
            Returns a pd.DataFrame with confidence intervalls, by default False
        alpha : float, optional
            Alpha level for confidence intervalls, by default DEFAULT_ALPHA

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
            df, X = _merge_X(df, X)

        # don't compute confidence intervals if not asked for
        with self._return_pred_int(return_pred_int):
            out = self._forecaster.predict(df)

        out.set_index("ds", inplace=True)
        y_pred = out.loc[:, "yhat"]

        if return_pred_int:
            pred_int = out.loc[:, ["yhat_lower", "yhat_upper"]]
            pred_int.columns = pred_int.columns.str.strip("yhat_")
            return y_pred, pred_int
        else:
            return y_pred

    def get_fitted_params(self):
        """Get fitted parameters

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
        """Checking arguments for changepoints and assign related arguments

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

    @contextmanager
    def _return_pred_int(self, return_pred_int):
        if not return_pred_int:
            # setting uncertainty samples to zero avoids computing pred ints
            self._forecaster.uncertainty_samples = 0
        try:
            yield
        finally:
            if not return_pred_int:
                self._forecaster.uncertainty_samples = self.uncertainty_samples


def _merge_X(df, X):
    """Merge X and df on the DatetimeIndex

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
    if "ds" in X.columns:
        raise ValueError("Column name 'ds' is reserved in fbprophet")
    X.loc[:, "ds"] = X.index
    # df = df.merge(X, how="inner", on="ds", copy=False)
    df = df.merge(X, how="inner", on="ds")
    return df, X.drop(columns="ds")


class _suppress_stdout_stderr(object):
    """
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
