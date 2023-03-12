# -*- coding: utf-8 -*-
"""Implements VAR Model as interface to statsmodels."""

__all__ = ["VAR"]
__author__ = ["thayeylolu", "aiwalter", "lbventura"]

import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class VAR(_StatsModelsAdapter):
    """
    A VAR model is a generalisation of the univariate autoregressive.

    Direct interface for `statsmodels.tsa.vector_ar`
    A model for forecasting a vector of time series[1].

    Parameters
    ----------
    maxlags: int or None (default=None)
        Maximum number of lags to check for order selection,
        defaults to 12 * (nobs/100.)**(1./4)
    method : str (default="ols")
        Estimation method to use
    verbose : bool (default=False)
        Print order selection output to the screen
    trend : str {"c", "ct", "ctt", "n"} (default="c")
        "c" - add constant
        "ct" - constant and trend
        "ctt" - constant, linear and quadratic trend
        "n" - co constant, no trend
        Note that these are prepended to the columns of the dataset.
    missing: str, optional (default='none')
        A string specifying if data is missing
    freq: str, tuple, datetime.timedelta, DateOffset or None, optional (default=None)
        A frequency specification for either `dates` or the row labels from
        the endog / exog data.
    dates: array_like, optional (default=None)
        An array like object containing dates.
    ic: One of {'aic', 'fpe', 'hqic', 'bic', None} (default=None)
        Information criterion to use for VAR order selection.
        aic : Akaike
        fpe : Final prediction error
        hqic : Hannan-Quinn
        bic : Bayesian a.k.a. Schwarz
    random_state : int, RandomState instance or None, optional ,
        default=None – If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    References
    ----------
    [1] Athanasopoulos, G., Poskitt, D. S., & Vahid, F. (2012).
    Two canonical VARMA forms: Scalar component models vis-à-vis the echelon form.
    Econometric Reviews, 31(1), 60–83, 2012.

    Examples
    --------
    >>> from sktime.forecasting.var import VAR
    >>> from sktime.datasets import load_longley
    >>> _, y = load_longley()
    >>> forecaster = VAR()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    VAR(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _fitted_param_names = ("aic", "fpe", "hqic", "bic")

    _tags = {
        "scitype:y": "multivariate",
        "y_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "univariate-only": False,
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        maxlags=None,
        method="ols",
        verbose=False,
        trend="c",
        missing="none",
        dates=None,
        freq=None,
        ic=None,
        random_state=None,
    ):
        # Model params
        self.trend = trend
        self.maxlags = maxlags
        self.method = method
        self.verbose = verbose
        self.missing = missing
        self.dates = dates
        self.freq = freq
        self.ic = ic

        super(VAR, self).__init__(random_state=random_state)

    def _fit_forecaster(self, y, X=None):
        """Fit forecaster to training data.

        Wraps Statsmodel's VAR fit method.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        from statsmodels.tsa.api import VAR as _VAR

        self._forecaster = _VAR(
            endog=y, exog=X, dates=self.dates, freq=self.freq, missing=self.missing
        )
        self._fitted_forecaster = self._forecaster.fit(
            trend=self.trend,
            maxlags=self.maxlags,
            method=self.method,
            verbose=self.verbose,
            ic=self.ic,
        )
        return self

    def _predict(self, fh, X=None):
        """
        Wrap Statmodel's VAR forecast method.

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
        y_pred_outsample = None
        y_pred_insample = None
        exog_future = X.values if X is not None else None
        # fh in stats
        # fh_int = fh.to_absolute_int(self._y.index[0], self._y.index[-1])
        fh_int = fh.to_relative(self.cutoff)
        n_lags = self._fitted_forecaster.k_ar

        # out-sample predictions
        if fh_int.max() > 0:
            y_pred_outsample = self._fitted_forecaster.forecast(
                y=self._y.values[-n_lags:],
                steps=fh_int[-1],
                exog_future=exog_future,
            )
        # in-sample prediction by means of residuals
        if fh_int.min() <= 0:
            y_pred_insample = self._y - self._fitted_forecaster.resid
            y_pred_insample = y_pred_insample.values

        if y_pred_insample is not None and y_pred_outsample is not None:
            y_pred = np.concatenate([y_pred_outsample, y_pred_insample], axis=0)
        else:
            y_pred = (
                y_pred_insample if y_pred_insample is not None else y_pred_outsample
            )

        index = fh.to_absolute(self.cutoff)
        index.name = self._y.index.name
        y_pred = pd.DataFrame(
            y_pred[fh.to_indexer(self.cutoff), :],
            index=fh.to_absolute(self.cutoff),
            columns=self._y.columns,
        )
        return y_pred

    def _predict_interval(self, fh, X=None, coverage: [float] = None):
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
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        model = self._fitted_forecaster
        fh_int = fh.to_relative(self.cutoff)
        steps = fh_int[-1]
        n_lags = model.k_ar

        y_cols_no_space = [str(col).replace(" ", "") for col in self._y.columns]

        df_list = []

        for cov in coverage:

            alpha = 1 - cov

            fcast_interval = model.forecast_interval(
                self._y.values[-n_lags:], steps=steps, alpha=alpha
            )
            lower_int, upper_int = fcast_interval[1], fcast_interval[-1]

            lower_df = pd.DataFrame(
                lower_int,
                columns=[
                    col + " " + str(alpha) + " " + "lower" for col in y_cols_no_space
                ],
            )
            upper_df = pd.DataFrame(
                upper_int,
                columns=[
                    col + " " + str(alpha) + " " + "upper" for col in y_cols_no_space
                ],
            )

            df_list.append(pd.concat((lower_df, upper_df), axis=1))

        concat_df = pd.concat(df_list, axis=1)

        concat_df_columns = list(
            OrderedDict.fromkeys(
                [
                    col_df
                    for col in y_cols_no_space
                    for col_df in concat_df.columns
                    if col in col_df
                ]
            )
        )

        pre_output_df = concat_df[concat_df_columns]

        pre_output_df_2 = pd.DataFrame(
            pre_output_df.values,
            columns=pd.MultiIndex.from_tuples(
                [col.split(" ") for col in pre_output_df]
            ),
        )

        final_columns = list(
            itertools.product(
                *[
                    self._y.columns,
                    coverage,
                    pre_output_df_2.columns.get_level_values(2).unique(),
                ]
            )
        )

        final_df = pd.DataFrame(
            pre_output_df_2.iloc[fh.to_indexer(self.cutoff), :].values,
            columns=pd.MultiIndex.from_tuples(final_columns),
        )

        final_df.index = fh.to_absolute(self.cutoff)
        final_df.index.name = self._y.index.name

        return final_df

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
        params1 = {"maxlags": 3}

        params2 = {"trend": "ctt"}  # breaks with "ic": "aic"}, see #4055

        return [params1, params2]
