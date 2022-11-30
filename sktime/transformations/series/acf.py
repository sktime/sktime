#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Auto-correlation transformations.

Module :mod:`sktime.transformations.series` implements auto-correlation
transformers.
"""

__author__ = ["afzal442"]
__all__ = ["AutoCorrelationTransformer", "PartialAutoCorrelationTransformer"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class AutoCorrelationTransformer(BaseTransformer):
    """Auto-correlation transformer.

    The autocorrelation function measures how correlated a timeseries is
    with itself at different lags. The AutocorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Parameters
    ----------
    adjusted : bool, default=False
        If True, then denominators for autocovariance are n-k, otherwise n.

    n_lags : int, default=None
        Number of lags to return autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs - 1).

    fft : bool, default=False
        If True, computes the ACF via FFT.

    missing : {"none", "raise", "conservative", "drop"}, default="none"
        How missing values are to be treated in autocorrelation function
        calculations.

        - "none" performs no checks or handling of missing values
        - “raise” raises an exception if NaN values are found.
        - “drop” removes the missing observations and then estimates the
          autocovariances treating the non-missing as contiguous.
        - “conservative” computes the autocovariance using nan-ops so that nans
          are removed when computing the mean and cross-products that are used to
          estimate the autocovariance. "n" in calculation is set to the number of
          non-missing observations.

    See Also
    --------
    PartialAutoCorrelationTransformer

    Notes
    -----
    Provides wrapper around statsmodels
    `acf <https://www.statsmodels.org/devel/generated/
    statsmodels.tsa.stattools.acf.html>`_ function.

    Examples
    --------
    >>> from sktime.transformations.series.acf import AutoCorrelationTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = AutoCorrelationTransformer(n_lags=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": True,
        "fit_is_empty": True,
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        adjusted=False,
        n_lags=None,
        fft=False,
        missing="none",
    ):
        self.adjusted = adjusted
        self.n_lags = n_lags
        self.fft = fft
        self.missing = missing
        super(AutoCorrelationTransformer, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        from statsmodels.tsa.stattools import acf

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = acf(
            X,
            adjusted=self.adjusted,
            nlags=self.n_lags,
            qstat=False,
            fft=self.fft,
            alpha=None,
            missing=self.missing,
        )
        return pd.Series(zt)

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
        return [{}, {"n_lags": 1}]


class PartialAutoCorrelationTransformer(BaseTransformer):
    """Partial auto-correlation transformer.

    The partial autocorrelation function measures the conditional correlation
    between a timeseries and its self at different lags. In particular,
    the correlation between a time period and a lag, is calculated conditional
    on all the points between the time period and the lag.

    The PartialAutoCorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Parameters
    ----------
    n_lags : int, default=None
        Number of lags to return partial autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs // 2 - 1).

    method : str, default="ywadjusted"
        Specifies which method for the calculations to use.

        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in
          denominator for acovf. Default.
        - "ywm" or "ywmle" : Yule-Walker without adjustment.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
          common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
          adjustment.
        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias
          correction.
        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias
          correction.

    See Also
    --------
    AutoCorrelationTransformer

    Notes
    -----
    Provides wrapper around statsmodels
    `pacf <https://www.statsmodels.org/devel/generated/
    statsmodels.tsa.stattools.pacf.html>`_ function.


    Examples
    --------
    >>> from sktime.transformations.series.acf import PartialAutoCorrelationTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = PartialAutoCorrelationTransformer(n_lags=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": True,
        "fit_is_empty": True,
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_lags=None,
        method="ywadjusted",
    ):
        self.n_lags = n_lags
        self.method = method
        super(PartialAutoCorrelationTransformer, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        from statsmodels.tsa.stattools import pacf

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = pacf(X, nlags=self.n_lags, method=self.method, alpha=None)
        return pd.Series(zt)

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
        return [{}, {"n_lags": 1}]
