# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implements Generalized Autoregressive Conditional Heteroskedasticity models."""

__author__ = ["Vasudeva-bit"]
__all__ = ["ARCH"]

import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.stats as stats

from sktime.forecasting.base import BaseForecaster


class ARCH(BaseForecaster):
    r"""Directly interfaces ARCH models from python package arch.

    ARCH models are a popular class of volatility models that use observed values of
    returns or residuals as volatility shocks to forecast the volatility in high
    frequency time series data..

    A complete ARCH model is divided into three components:
        a mean model, e.g., a constant mean or an ARX;
        a volatility process, e.g., a GARCH or an EGARCH process; and
        a distribution for the standardized residuals.

    Parameters
    ----------
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
        'Zero', 'LS', 'AR', 'ARX', 'HAR' and  'HARX'
    lags : int or list (int), optional
        Either a scalar integer value indicating lag length or a list of
        integers specifying lag locations.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIARCH' and 'HARCH'
    p : int, optional
        Lag order of the symmetric innovation
    o : int, optional
        Lag order of the asymmetric innovation
    q : int, optional
        Lag order of lagged volatility or equivalent
    power : float, optional
        Power to use with GARCH and related models
    dist : int, optional
        Name of the error distribution.  Currently supported options are:
            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"
    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    rescale : bool
        Flag indicating whether to automatically rescale data if the scale
        of the data is likely to produce convergence issues when estimating
        model parameters. If False, the model is estimated on the data without
        transformation.  If True, than y is rescaled and the new scale is
        reported in the estimation results.
    update_freq : int, optional
        Frequency of iteration updates.  Output is generated every
        update_freq iterations. Set to 0 to disable iterative output.
    disp : {bool, "off", "final"}
        Either 'final' to print optimization result or 'off' to display
        nothing. If using a boolean, False is "off" and True is "final"
    starting_values : np.ndarray, optional
        Array of starting values to use.  If not provided, starting values
        are constructed by the model components.
    cov_type : str, optional
        Estimation method of parameter covariance.  Supported options are
        'robust', which does not assume the Information Matrix Equality
        holds and 'classic' which does.  In the ARCH literature, 'robust'
        corresponds to Bollerslev-Wooldridge covariance estimator.
    show_warning : bool, optional
        Flag indicating whether convergence warnings should be shown
    first_obs : {int, str, datetime, Timestamp}
        First observation to use when estimating model
    last_obs : {int, str, datetime, Timestamp}
        Last observation to use when estimating model
    tol : float, optional
        Tolerance for termination.
    options : dict, optional
        Options to pass to scipy.optimize.minimize.  Valid entries
        include 'ftol', 'eps', 'disp', and 'maxiter'.
    backcast : {float, np.ndarray}, optional
        Value to use as backcast. Should be measure \\sigma^2_0
        since model-specific non-linear transformations are applied to
        value before computing the variance recursions.
    params : {np.ndarray, Series}
        Parameters required to forecast. Must be identical in shape to the
        parameters computed by fitting the model.
    start : {int, datetime, Timestamp, str}, optional
        An integer, datetime or str indicating the first observation to produce the
        forecast for. Datetimes can only be used with pandas inputs that have a
        datetime index. Strings must be convertible to a date time, such as in
        '1945-01-01'.
    align : str, optional
        Either 'origin' or 'target'. When set of 'origin', the t-th row of forecast
        contains the forecasts for t+1, t+2, ..., t+h. When set to 'target', the
        t-th row contains the 1-step ahead forecast from time t-1, the 2 step from
        time t-2, ..., and the h-step from time t-h. 'target' simplified computing
        forecast errors since the realization and h-step forecast are aligned.
    method : {'analytic', 'simulation', 'bootstrap'}
        Method to use when producing the forecast. The default is analytic.
        The method only affects the variance forecast generation. Not all volatility
        models support all methods.
        In particular, volatility models that do not evolve in squares such as
        EGARCH or TARCH
        do not support the 'analytic' method for horizons > 1.
    simulations : int
        Number of simulations to run when computing the forecast using either
        simulation or bootstrap.
    rng : callable, optional
        Custom random number generator to use in simulation-based forecasts.
        Must produce random samples using the syntax rng(size) where size the
        2-element tuple (simulations, horizon).
    random_state : RandomState, optional
        NumPy RandomState instance to use when method is 'bootstrap'
    reindex : bool, optional
        Whether to reindex the forecasts to have the same dimension as the series
        being forecast. Prior to 4.18 this was the default. As of 4.19 this is now
        optional. If not provided, a warning is raised about the future change in
        the default which will occur after September 2021.

    See Also
    --------
    StatsForecastARCH
    StatsForecastGARCH

    References
    ----------
    .. [1] GitHub repository of arch package (soft dependency).
       https://github.com/bashtage/arch
    .. [2] Documentation of arch package (soft dependency). Forecasting Volatility with
       ARCH and it's variants.
       https://arch.readthedocs.io/en/latest/univariate/introduction.html

    Examples
    --------
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>> from sktime.forecasting.arch import ARCH  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = ARCH()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    ARCH(...)
    >>> y_pred = forecaster.predict(fh=1)  # doctest: +SKIP
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": "arch",
        "capability:pred_int": True,
        "ignores-exogeneous-X": True,
    }

    def __init__(
        self,
        mean="Constant",
        lags=0,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        power=2.0,
        dist="Normal",
        hold_back=None,
        rescale=False,
        update_freq=0,
        disp=False,
        starting_values=None,
        cov_type="robust",
        show_warning=False,
        first_obs=None,
        last_obs=None,
        tol=None,
        options=None,
        backcast=None,
        params=None,
        start=None,
        align="origin",
        method="simulation",
        simulations=10,
        rng=None,
        random_state=None,
        reindex=False,
    ):
        self.mean = mean
        self.lags = lags
        self.vol = vol
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.dist = dist
        self.hold_back = hold_back
        self.rescale = rescale
        self.update_freq = update_freq
        self.disp = disp
        self.starting_values = starting_values
        self.cov_type = cov_type
        self.show_warning = show_warning
        self.first_obs = first_obs
        self.last_obs = last_obs
        self.tol = tol
        self.options = options
        self.backcast = backcast
        self.params = params
        self.start = start
        self.align = align
        self.method = method
        self.simulations = simulations
        self.rng = rng
        self.random_state = random_state
        self.reindex = reindex

        if self.mean in ["ARX", "HARX"]:
            self.set_tags(**{"ignores-exogeneous-X": False})

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the training data to the estimator.

        Parameters
        ----------
        y : pd.Series
            Time series to fit to.
        X : pd.DataFrame (default=None)
            Exogenous time series.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the positive steps ahead to predict.

        Returns
        -------
        self : returns an instance of self
        """
        from arch import arch_model as _ARCH
        from arch.__future__ import reindexing

        reindexing._warn_for_future_reindexing = False

        if fh:
            self._horizon = fh

        self._forecaster = _ARCH(
            y=y,
            x=X,
            mean=self.mean,
            lags=self.lags,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            power=self.power,
            dist=self.dist,
            hold_back=self.hold_back,
            rescale=self.rescale,
        )
        self._fitted_forecaster = self._forecaster.fit(
            update_freq=self.update_freq,
            disp=self.disp,
            starting_values=self.starting_values,
            cov_type=self.cov_type,
            show_warning=self.show_warning,
            first_obs=self.first_obs,
            last_obs=self.last_obs,
            tol=self.tol,
            options=self.options,
            backcast=self.backcast,
        )
        return self

    def _get_arch_result_object(self, fh=None, X=None):
        """Return ARCH result object.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the positive steps ahead to predict.
        X : optional (default=None)
            (default=None) Exogenous time series.

        Returns
        -------
        ArchResultObject : ARCH result object, full_range, abs_idx in a tuple
            mean, variance forecasts, full_range, abs_idx
        """
        if fh:
            self._horizon = fh

        abs_idx = self._horizon.to_absolute_int(self._y.index[0], self.cutoff)
        start, end = abs_idx[[0, -1]]
        start = min(start, len(self._y))

        if X is not None:
            x = {}
            for col in X.columns:
                x[str(col)] = np.array(X[col])
        else:
            x = None

        ArchResultObject = self._fitted_forecaster.forecast(
            x=x,
            horizon=end - start + 1,
            params=self.params,
            start=self.start,
            align=self.align,
            method=self.method,
            simulations=self.simulations,
            rng=self.rng,
            random_state=self.random_state,
            reindex=self.reindex,
        )
        full_range = pd.RangeIndex(start=start, stop=end + 1)

        return (ArchResultObject, full_range, abs_idx)

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the positive steps ahead to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame (default=None)
            Exogenous time series.

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        ArchResultObject, full_range, abs_idx = self._get_arch_result_object(fh=fh, X=X)
        y_pred = pd.Series(
            ArchResultObject.mean.values[-1],
            index=full_range,
            name=str(self._y.name),
        )
        y_pred = y_pred.loc[abs_idx.to_pandas()]
        y_pred.index = self._horizon.to_absolute_index(self.cutoff)

        return y_pred

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction intervals for a forecast.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
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
        ArchResultObject, full_range, abs_idx = self._get_arch_result_object(fh=fh, X=X)
        std_err = np.sqrt(np.array(ArchResultObject.variance.values[-1]))
        mean_forecast = np.array(ArchResultObject.mean.values[-1])

        y_col_name = self._y.name
        df_list = []
        for confidence in coverage:
            alpha = 1 - confidence
            z_critical = stats.norm.ppf(1 - (alpha) / 2)
            lower_int = mean_forecast - (z_critical * std_err)
            upper_int = mean_forecast + (z_critical * std_err)
            lower_df = pd.DataFrame(
                lower_int,
                columns=[y_col_name + " " + str(alpha) + " " + "lower"],
            )
            upper_df = pd.DataFrame(
                upper_int,
                columns=[y_col_name + " " + str(alpha) + " " + "upper"],
            )
            df_list.append(pd.concat((lower_df, upper_df), axis=1))
        concat_df = pd.concat(df_list, axis=1)
        concat_df_columns = list(
            OrderedDict.fromkeys(
                [
                    col_df
                    for col in y_col_name
                    for col_df in concat_df.columns
                    if col in col_df
                ]
            )
        )
        df = concat_df[concat_df_columns]
        df = pd.DataFrame(
            df.values,
            columns=pd.MultiIndex.from_tuples([col.split(" ") for col in df]),
            index=full_range,
        )
        final_columns = list(
            itertools.product(
                *[
                    [y_col_name],
                    coverage,
                    df.columns.get_level_values(2).unique(),
                ]
            )
        )
        df = pd.DataFrame(
            df.loc[abs_idx.to_pandas()].values,
            columns=pd.MultiIndex.from_tuples(final_columns),
        )
        df.index = self._horizon.to_absolute_index(self.cutoff)
        return df

    def _predict_var(self, fh=None, X=None, cov=False):
        """Compute/return variance forecasts.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the positive steps ahead to predict.
        X : optional (default=None)
            (default=None) Exogenous time series.

        Returns
        -------
        pred_var : pd.DataFrame
            Variance forecasts
        """
        ArchResultObject, full_range, abs_idx = self._get_arch_result_object(fh=fh, X=X)
        pred_var = pd.Series(
            ArchResultObject.variance.values[-1],
            index=full_range,
            name=self._y.name,
        )
        pred_var = pred_var.loc[abs_idx.to_pandas()]
        pred_var.index = self._horizon.to_absolute_index(self.cutoff)

        return pred_var

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        fitted_params = dict(self._fitted_forecaster.params)
        return fitted_params

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {
            "mean": "Constant",
            "lags": 0,
            "vol": "GARCH",
            "p": 1,
            "o": 0,
            "q": 1,
            "power": 2.0,
            "dist": "Normal",
            "hold_back": None,
            "rescale": False,
        }
        params2 = {
            "mean": "ARX",
            "vol": "ARCH",
            "p": 1,
            "dist": "normal",
            "rescale": False,
        }
        return [params1, params2]

    def summary(self):
        """Summary of the fitted model."""
        self.check_is_fitted()
        return self._fitted_forecaster.summary()
