# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Granger causality parameter estimator."""

__author__ = ["Spinachboul"]
__all__ = ["GrangerCausalityFitter"]

import pandas as pd

from sktime.param_est.base import BaseParamFitter
from sktime.utils.dependencies._dependencies import _check_soft_dependencies


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
        # Check for statsmodels dependency
        _check_soft_dependencies("statsmodels", severity="error")

        # Import statsmodels functions only when needed
        from statsmodels.tsa.stattools import (
            adfuller,
            coint,
            grangercausalitytests,
            kpss,
            pacf,
            range_unit_root_test,
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if X.shape[1] == 1:  # the number of columns in the input data
            X_bivariate = X.copy()
            col_name = X.columns[0]
            X_bivariate[f"{col_name}_lagged"] = X_bivariate[col_name].shift(1)
            X_bivariate = X_bivariate.dropna()  # Fixed typo: was X.bivariate
            X = X_bivariate

        elif X.shape[1] != 2:
            raise ValueError("Input X must have exactly two columns.")

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise ValueError("All columns in X must be numeric.")

        # Perform tests
        self.stationarity_ = {
            col: {
                "ADF": self.adf_test(X[col], adfuller),
                "KPSS": self.kpss_test(X[col], kpss),
                "RUR": self.range_unit_root(X[col], range_unit_root_test),
            }
            for col in X.columns
        }
        self.cointegration_ = self.cointegration_test(X.iloc[:, 0], X.iloc[:, 1], coint)
        self.pacf_ = {col: self.pacf_analysis(X[col], pacf) for col in X.columns}
        granger_result = self.run_granger_test(X, grangercausalitytests)
        self.best_lag_ = granger_result["best_lag"]
        self.best_pvalue_ = granger_result["best_pvalue"]

        self._is_fitted = True
        return self

    def adf_test(self, series, adfuller_func):
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Parameters
        ----------
        series : pd.Series
            The time series to test for stationarity.
        adfuller_func : function
            The adfuller function from statsmodels.

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
        result = adfuller_func(
            series, maxlag=self.maxlag, regression="c", autolag="AIC", regresults=False
        )
        return {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "critical values": result[4],
        }

    def kpss_test(self, series, kpss_func):
        """
        Perform KPSS test for stationarity.

        Parameters
        ----------
        series : pd.Series
            The time series to test for stationarity.
        kpss_func : function
            The kpss function from statsmodels.

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
        statistic, p_value, lags, critical_values = kpss_func(
            series, regression="c", nlags="auto", store=False
        )
        return {
            "KPSS Statistic": statistic,
            "p-value": p_value,
            "lags": lags,
            "critical values": critical_values,
        }

    def range_unit_root(self, series, range_unit_root_func):
        """
        Perform Range Unit Root Test for stationarity.

        Parameters
        ----------
        series : pd.Series
            The time series to test for stationarity.
        range_unit_root_func : function
            The range_unit_root_test function from statsmodels.

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
        stat, p_value, crit, rstore = range_unit_root_func(series, store=False)
        return {
            "RUR Statistic": stat,
            "p-value": p_value,
            "critical values": crit,
            "rstore": rstore,
        }

    def cointegration_test(self, y0, y1, coint_func):
        """
        Test for cointegration between two time series.

        Parameters
        ----------
        y0 : pd.Series
            First time series.
        y1 : pd.Series
            Second time series.
        coint_func : function
            The coint function from statsmodels.

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
        stat, p_value, crit = coint_func(
            y0, y1, trend="c", method="aeg", maxlag=self.maxlag, autolag="AIC"
        )
        return {
            "Cointegration Statistic": stat,
            "p-value": p_value,
            "critical values": crit,
        }

    def pacf_analysis(self, series, pacf_func, nlags=30, method="ywadjusted"):
        """
        Calculate partial autocorrelation function (PACF) for a time series.

        Parameters
        ----------
        series : pd.Series
            The time series to analyze.
        pacf_func : function
            The pacf function from statsmodels.
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
        pacf_vals, confint = pacf_func(series, nlags=nlags, method=method, alpha=0.05)
        return {"PACF": pacf_vals, "Confidence Interval": confint}

    def run_granger_test(self, data, grangercausalitytests_func):
        """
        Run Granger causality tests and find the optimal lag order.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with two time series columns.
        grangercausalitytests_func : function
            The grangercausalitytests function from statsmodels.

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
        result = grangercausalitytests_func(
            data, maxlag=self.maxlag, verbose=self.verbose, addconst=self.addconst
        )
        best_ic = float("inf")
        best_lag = None
        best_pvalue = None

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
