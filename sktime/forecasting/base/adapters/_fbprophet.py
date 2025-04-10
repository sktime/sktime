# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for Facebook prophet to be used in sktime framework."""

__author__ = ["mloning", "aiwalter", "fkiraly"]
__all__ = ["_ProphetAdapter"]

import os

import pandas as pd

from sktime.forecasting.base import BaseForecaster


class _ProphetAdapter(BaseForecaster):
    """Base class for interfacing prophet and neuralprophet."""

    _tags = {
        "authors": ["bletham", "tcuongd", "mloning", "aiwalter", "fkiraly"],
        # bletham and tcuongd for prophet/fbprophet
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": True,
        "y_inner_mtype": "pd.DataFrame",
        "python_dependencies": "prophet",
    }

    def _convert_int_to_date(self, y):
        """Convert int to date, for use by prophet."""
        y = y.copy()
        idx_max = y.index[-1] + 1
        int_idx = pd.date_range(start="2000-01-01", periods=idx_max, freq="D")
        int_idx = int_idx[y.index]
        y.index = int_idx
        return y

    def _convert_input_to_date(self, y):
        """Coerce y.index to pd.DatetimeIndex, for use by prophet."""
        if y is None:
            return None
        elif type(y.index) is pd.PeriodIndex:
            y = y.copy()
            y.index = y.index.to_timestamp()
        elif pd.api.types.is_integer_dtype(y.index):
            y = self._convert_int_to_date(y)
        # else y is pd.DatetimeIndex as prophet expects, and needs no conversion
        return y

    def _remember_y_input_index_type(self, y):
        """Remember input type of y by setting attributes, for use in _fit."""
        self.y_index_was_period_ = type(y.index) is pd.PeriodIndex
        self.y_index_was_int_ = pd.api.types.is_integer_dtype(y.index)

    def _fit(self, y, X, fh):
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

        # sets y_index_was_period_ and self.y_index_was_int_ flags
        # to remember the index type of y before conversion
        self._remember_y_input_index_type(y)

        # various type input indices are converted to datetime
        # since facebook prophet can only deal with dates
        y = self._convert_input_to_date(y)
        X = self._convert_input_to_date(X)

        # We have to bring the data into the required format for fbprophet
        # the index should not be pandas index, but in a column named "ds"
        df = y.copy()
        df.columns = ["y"]
        df.index.name = "ds"
        df = df.reset_index()

        # Add seasonality/seasonalities and collect condition names
        condition_names = []
        if self.add_seasonality:
            if isinstance(self.add_seasonality, dict):
                self._forecaster.add_seasonality(**self.add_seasonality)
                if (
                    condition_name := self.add_seasonality.get("condition_name", None)
                ) is not None:
                    condition_names.append(condition_name)
            elif isinstance(self.add_seasonality, list):
                for seasonality in self.add_seasonality:
                    self._forecaster.add_seasonality(**seasonality)
                    if (
                        condition_name := seasonality.get("condition_name", None)
                    ) is not None:
                        condition_names.append(condition_name)

        # Add country holidays
        if self.add_country_holidays:
            self._forecaster.add_country_holidays(**self.add_country_holidays)

        # Add regressor (multivariate)
        if X is not None:
            X = X.copy()
            df, X = _merge_X(df, X)
            regressor_names = (col for col in X.columns if col not in condition_names)
            for col in regressor_names:
                self._forecaster.add_regressor(col)

        # Add floor and bottom when growth is logistic
        if self.growth == "logistic":
            if self.growth_cap is None:
                raise ValueError(
                    "Since `growth` param is set to 'logistic', expecting `growth_cap`"
                    " to be non `None`: a float."
                )

            df["cap"] = self.growth_cap
            df["floor"] = self.growth_floor

        if hasattr(self, "fit_kwargs") and isinstance(self.fit_kwargs, dict):
            fit_kwargs = self.fit_kwargs
        else:
            fit_kwargs = {}
        if self.verbose:
            self._forecaster.fit(df=df, **fit_kwargs)
        else:
            with _suppress_stdout_stderr():
                self._forecaster.fit(df=df, **fit_kwargs)

        return self

    def _get_prophet_fh(self):
        """Get a prophet compatible fh, in datetime, even if fh was int."""
        fh = self.fh.to_absolute_index(cutoff=self.cutoff)
        if isinstance(fh, pd.PeriodIndex):
            fh = fh.to_timestamp()
        if not isinstance(fh, pd.DatetimeIndex):
            max_int = fh[-1] + 1
            fh_date = pd.date_range(start="2000-01-01", periods=max_int, freq="D")
            fh = fh_date[fh]
        return fh

    def _convert_X_for_exog(self, X, fh):
        """Conerce index of X to index expected by prophet."""
        if X is None:
            return None
        elif isinstance(X.index, pd.PeriodIndex):
            X = X.copy()
            X = X.loc[self.fh.to_absolute_index(self.cutoff)]
            X.index = X.index.to_timestamp()
        elif pd.api.types.is_integer_dtype(X.index):
            X = X.copy()
            X = X.loc[self.fh.to_absolute(self.cutoff).to_numpy()]
            X.index = fh
        # else X is pd.DatetimeIndex as prophet expects, and needs no conversion
        else:
            X = X.loc[fh]
        return X

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
        fh = self._get_prophet_fh()
        df = pd.DataFrame({"ds": fh}, index=fh)

        X = self._convert_X_for_exog(X, fh)

        # Merge X with df (of created future DatetimeIndex values)
        if X is not None:
            df, X = _merge_X(df, X)

        if self.growth == "logistic":
            df["cap"] = self.growth_cap
            df["floor"] = self.growth_floor

        out = self._forecaster.predict(df)

        out.set_index("ds", inplace=True)
        y_pred = out.loc[:, "yhat"]

        # bring outputs into required format
        # same column names as training data, index should be index, not "ds"
        y_pred = pd.DataFrame(y_pred)
        y_pred.reset_index(inplace=True)
        y_pred.index = y_pred["ds"].values
        y_pred.drop("ds", axis=1, inplace=True)
        y_pred.columns = self._y.columns

        if self.y_index_was_int_ or self.y_index_was_period_:
            y_pred.index = self.fh.to_absolute_index(cutoff=self.cutoff)

        return y_pred

    def _predict_interval(self, fh, X, coverage):
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
        fh = self._get_prophet_fh()

        X = self._convert_X_for_exog(X, fh)

        # prepare the return DataFrame - empty with correct cols
        var_names = self._get_varnames()
        var_name = var_names[0]

        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx)

        # prepare the DataFrame to pass to prophet
        df = pd.DataFrame({"ds": fh}, index=fh)
        if X is not None:
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
            #  because prophet (erroneously?) uses MC independent for upper/lower
            #  so if coverage is small, it can happen that upper < lower in prophet
            pred_int[(var_name, c, "lower")] = out_prophet.min(axis=1)
            pred_int[(var_name, c, "upper")] = out_prophet.max(axis=1)

        if self.y_index_was_int_ or self.y_index_was_period_:
            pred_int.index = self.fh.to_absolute_index(cutoff=self.cutoff)

        return pred_int

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict

        References
        ----------
        https://facebook.github.io/prophet/docs/additional_topics.html
        """
        fitted_params = {}
        for name in ["k", "m", "sigma_obs"]:
            fitted_params[name] = self._forecaster.params[name].flatten()[0]
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
        Exogeneous data
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


class _suppress_stdout_stderr:
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
