# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Granger causality parameter estimator."""

__author__ = ["Spinachboul"]
__all__ = ["GrangerCausalityFitter"]

import pandas as pd

from sktime.exceptions import NotFittedError
from sktime.param_est.base import BaseParamFitter
from sktime.utils.dependencies._dependencies import _check_soft_dependencies

# Import statsmodels functions conditionally
_check_soft_dependencies("statsmodels", severity="warning")
try:
    from statsmodels.tsa.stattools import (
        adfuller,
        coint,
        grangercausalitytests,
        kpss,
        pacf,
        range_unit_root_test,
    )

    _statsmodels_available = True
except ImportError:
    _statsmodels_available = False


class GrangerCausalityFitter(BaseParamFitter):
    """Estimates optimal lag for Granger causality tests.

    Parameters
    ----------
    maxlag : int, default=12
        Maximum number of lags to test.
    ic : str, default="bic"
        Information criterion to use for lag selection ("aic" or "bic").
    verbose : bool, default=False
        If True, prints the results of each lag test.
    addconst : bool, default=True
        Include a constant in the model.

    For more information on the parameters, see the documentation for
    `statsmodels.tsa.stattools.grangercausalitytests`.

    Notes
    -----
    This estimator implements Granger causality testing using the statsmodels library.
    Granger causality is a statistical concept that tests whether one time series is
    useful in forecasting another. This class also provides additional time series
    testing capabilities for stationarity (ADF, KPSS, range unit root tests),
    cointegration, and partial autocorrelation.

    References
    ----------
    .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
    """

    __tags = {
        "X_inner_mtype": "pd.DataFrame",
        "scitype:transform-input": "Series",
        "capability:missing_values": False,
        "capability:multivariate": True,
        "requires_y": False,
        "X_inner_X_required_cols": 2,
        "authors": "Spinachboul",
        "python_dependencies": "statsmodels",
    }

    def __init__(self, maxlag=12, ic="bic", verbose=False, addconst=True):
        """Initialize parameters."""
        self.maxlag = maxlag
        self.ic = ic
        self.verbose = verbose
        self.addconst = addconst
        self._is_fitted = False
        super().__init__()

        # Check if statsmodels is available during initialization
        if not _statsmodels_available:
            import warnings

            warnings.warn(
                "The 'statsmodels' package is required for GrangerCausalityFitter. "
                "Please install it with: pip install statsmodels",
                UserWarning,
            )

    def _fit(self, X):
        """
        Fit the Granger causality model and run related statistical tests.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with two time series columns.

        Returns
        -------
        self : reference to self
            Returns the fitted estimator.

        Raises
        ------
        TypeError
            If X is not a pandas DataFrame.
        ValueError
            If X does not have exactly two columns or if columns are not numeric.
        ModuleNotFoundError
            If statsmodels is not installed.

        Notes
        -----
        This method performs multiple tests on the provided time series data:
        - Stationarity tests (ADF, KPSS, Range Unit Root)
        - Cointegration test between the two series
        - Partial autocorrelation analysis
        - Granger causality tests to determine the optimal lag
        """
        # Check if statsmodels is available
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for GrangerCausalityFitter. "
                "Please install it with: pip install statsmodels"
            )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if X.shape[1] != 2:
            raise ValueError("Input X must have exactly two columns.")

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise ValueError("All columns in X must be numeric.")

        # Perform tests
        self.stationarity_ = {
            col: {
                "ADF": self.adf_test(X[col]),
                "KPSS": self.kpss_test(X[col]),
                "RUR": self.range_unit_root(X[col]),
            }
            for col in X.columns
        }
        self.cointegration_ = self.cointegration_test(X.iloc[:, 0], X.iloc[:, 1])
        self.pacf_ = {col: self.pacf_analysis(X[col]) for col in X.columns}
        granger_result = self.run_granger_test(X)
        self.best_lag_ = granger_result["best_lag"]
        self.best_pvalue_ = granger_result["best_pvalue"]

        self._is_fitted = True
        return self

    def _get_fitted_params(self):
        """
        Get the parameters learned by the estimator during fitting.

        Returns
        -------
        dict
            A dictionary containing:
            - 'best_lag': The optimal lag order determined by information criterion
            - 'best_pvalue': The p-value associated with the best lag
            - 'stationarity': Dictionary of stationarity test results for each column
            - 'cointegration': Results of cointegration test between columns
            - 'pacf': Partial autocorrelation function results for each column

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self._is_fitted:
            raise NotFittedError("The estimator has not been fitted yet.")

        return {
            "best_lag": self.best_lag_,
            "best_pvalue": self.best_pvalue_,
            "stationarity": self.stationarity_,
            "cointegration": self.cointegration_,
            "pacf": self.pacf_,
        }

    def adf_test(self, series):
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Parameters
        ----------
        series : pd.Series
            The time series to test for stationarity.

        Returns
        -------
        dict
            Dictionary containing:
            - 'ADF Statistic': Test statistic
            - 'p-value': p-value of the test
            - 'critical values': Critical values for different significance levels

        Notes
        -----
        The null hypothesis is that the series has a unit root (non-stationary).
        If the p-value is less than the significance level (typically 0.05),
        we can reject the null hypothesis and conclude that the series is stationary.

        References
        ----------
        .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
        """
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for this method. "
                "Please install it with: pip install statsmodels"
            )

        result = adfuller(
            series, maxlag=self.maxlag, regression="c", autolag="AIC", regresults=False
        )
        return {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "critical values": result[4],
        }

    def kpss_test(self, series):
        """
        Perform KPSS test for stationarity.

        Parameters
        ----------
        series : pd.Series
            The time series to test for stationarity.

        Returns
        -------
        dict
            Dictionary containing:
            - 'KPSS Statistic': Test statistic
            - 'p-value': p-value of the test
            - 'lags': Number of lags used
            - 'critical values': Critical values for different significance levels

        Notes
        -----
        The null hypothesis is that the series is stationary.
        If the p-value is less than the significance level (typically 0.05),
        we can reject the null hypothesis and conclude that the series is
        non-stationary. This test complements the ADF test as they have
        opposite null hypotheses.

        References
        ----------
        .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
        """
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for this method. "
                "Please install it with: pip install statsmodels"
            )

        statistic, p_value, lags, critical_values = kpss(
            series, regression="c", nlags="auto", store=False
        )
        return {
            "KPSS Statistic": statistic,
            "p-value": p_value,
            "lags": lags,
            "critical values": critical_values,
        }

    def range_unit_root(self, series):
        """
        Perform Range Unit Root Test for stationarity.

        Parameters
        ----------
        series : pd.Series
            The time series to test for stationarity.

        Returns
        -------
        dict
            Dictionary containing:
            - 'RUR Statistic': Test statistic
            - 'p-value': p-value of the test
            - 'critical values': Critical values for different significance levels
            - 'rstore': Additional test results

        Notes
        -----
        The Range Unit Root Test is another test for the presence of unit roots
        in time series data. The null hypothesis is that the series has a unit root
        (non-stationary).

        References
        ----------
        .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html
        """
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for this method. "
                "Please install it with: pip install statsmodels"
            )

        stat, p_value, crit, rstore = range_unit_root_test(series, store=False)
        return {
            "RUR Statistic": stat,
            "p-value": p_value,
            "critical values": crit,
            "rstore": rstore,
        }

    def cointegration_test(self, y0, y1):
        """
        Test for cointegration between two time series.

        Parameters
        ----------
        y0 : pd.Series
            First time series.
        y1 : pd.Series
            Second time series.

        Returns
        -------
        dict
            Dictionary containing:
            - 'Cointegration Statistic': Test statistic
            - 'p-value': p-value of the test
            - 'critical values': Critical values for different significance levels

        Notes
        -----
        Cointegration tests whether two non-stationary time series move together
        over time and are in a long-run equilibrium relationship. The test
        implemented is the Augmented Engle-Granger cointegration test.

        The null hypothesis is that the series are not cointegrated.
        If the p-value is less than the significance level (typically 0.05),
        we can reject the null hypothesis and conclude that the series are cointegrated.

        References
        ----------
        .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html
        """
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for this method. "
                "Please install it with: pip install statsmodels"
            )

        stat, p_value, crit = coint(
            y0, y1, trend="c", method="aeg", maxlag=self.maxlag, autolag="AIC"
        )
        return {
            "Cointegration Statistic": stat,
            "p-value": p_value,
            "critical values": crit,
        }

    def pacf_analysis(self, series, nlags=30, method="ywadjusted"):
        """
        Calculate partial autocorrelation function (PACF) for a time series.

        Parameters
        ----------
        series : pd.Series
            The time series to analyze.
        nlags : int, default=30
            Number of lags to include in the analysis.
        method : str, default='ywadjusted'
            Method to calculate the PACF. Options include 'yw', 'ywadjusted', 'ols',
            'ols-inefficient', 'ols-adjusted', and 'ld'.

        Returns
        -------
        dict
            Dictionary containing:
            - 'PACF': PACF values for each lag
            - 'Confidence Interval': 95% confidence intervals for each PACF value

        Notes
        -----
        The partial autocorrelation function (PACF) measures the correlation between
        observations that are k time periods apart, after controlling for the effects of
        intermediate observations. It is useful for identifying the appropriate order
        for an autoregressive (AR) model.

        References
        ----------
        .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html
        """
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for this method. "
                "Please install it with: pip install statsmodels"
            )

        pacf_vals, confint = pacf(series, nlags=nlags, method=method, alpha=0.05)
        return {"PACF": pacf_vals, "Confidence Interval": confint}

    def run_granger_test(self, data):
        """
        Run Granger causality tests and find the optimal lag order.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with two time series columns.

        Returns
        -------
        dict
            Dictionary containing:
            - 'best_lag': The best lag order based on specified information criterion
            - 'best_pvalue': The p-value associated with the best lag
            - 'full_result': Complete results from statsmodels grangercausalitytests

        Notes
        -----
        Granger causality tests whether a timer series helps forecasting another.
        The null hypothesis is that the first series does not Granger-cause the
        second series.

        This function runs tests for lags 1 to maxlag and identifies the best lag order
        based on the specified information criterion.

        References
        ----------
        .. [1] https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
        """
        if not _statsmodels_available:
            raise ModuleNotFoundError(
                "The 'statsmodels' package is required for this method. "
                "Please install it with: pip install statsmodels"
            )

        result = grangercausalitytests(
            data, maxlag=self.maxlag, verbose=self.verbose, addconst=self.addconst
        )
        best_ic = float("inf")
        best_lag = None

        for lag, res in result.items():
            ic_value = res[0][f"ssr_{self.ic}"][0]
            if ic_value < best_ic:
                best_ic = ic_value
                best_lag = lag
                best_pvalue = res[0]["ssr_ftest"][1]

        return {"best_lag": best_lag, "best_pvalue": best_pvalue, "full_result": result}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {"maxlag": 5, "ic": "bic"}
        params2 = {"maxlag": 10, "ic": "aic", "verbose": True, "addconst": False}

        return [params1, params2]
