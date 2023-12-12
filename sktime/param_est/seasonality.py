# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for seasonality."""

__author__ = ["fkiraly", "blazingbhavneek"]
__all__ = ["SeasonalityACF", "SeasonalityPeriodogram"]

import numpy as np

from sktime.param_est.base import BaseParamFitter


class SeasonalityACF(BaseParamFitter):
    """Find candidate seasonality parameter using autocorrelation function CI.

    Uses `statsmodels.tsa.stattools.act` for computing the autocorrelation function,
    and uses its testing functionality to determine candidate seasonality parameters.
    ("seasonality parameter" are integer lags, and abbreviated by sp, below)

    Obtains confidence intervals at a significance level, and returns lags
    with significant positive auto-correlation, ordered by lower confidence limit.

    Note: this should be applied to stationary series.
        Quick stationarity transformation can be achieved by differencing.
        See also: Differencer

    Parameters
    ----------
    candidate_sp : None, int or list of int, optional, default = None
        candidate sp to test, and to restrict tests to; ints must be 2 or larger
        if None, will test all integer lags between 2 and `nlags` (inclusive)
    p_threshold : float, optional, default=0.05
        significance threshold to apply in testing for seasonality
    adjusted : bool, optional, default=False
        If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional, default=None
        Number of lags to compute autocorrelations for and select from.
        At default None, uses `min(10 * np.log10(nobs), nobs - 1)`.
        Will be ignored if `candidate_sp` is provided.
    fft : bool, optional, default=True
        If True, computes the ACF via FFT.
    missing : str, ["none", "raise", "conservative", "drop"], optional, default="none"
        Specifies how NaNs are to be treated.
        "none" performs no checks.
        "raise" raises an exception if NaN values are found.
        "drop" removes the missing observations and treats non-missing as contiguous.
        "conservative" computes the autocovariance using nan-ops so that nans are
            removed when computing the mean and cross-products that are used to
            estimate the autocovariance. When using "conservative",
            n is set to the number of non-missing observations.

    Attributes
    ----------
    sp_ : int, seasonality period at lowest p-level, if any sub-threshold, else 1
        if `candidate_sp` is passed, will be in `candidate_sp` or 1
    sp_significant_ : list of int, seasonality periods with sub-threshold p-levels
        ordered increasingly by p-level. Empty list, not [1], if none are sub-threshold

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>>
    >>> X = load_airline().diff()[1:]  # doctest: +SKIP
    >>> sp_est = SeasonalityACF()  # doctest: +SKIP
    >>> sp_est.fit(X)  # doctest: +SKIP
    SeasonalityACF(...)
    >>> sp_est.get_fitted_params()["sp"]  # doctest: +SKIP
    12
    >>> sp_est.get_fitted_params()["sp_significant"]  # doctest: +SKIP
    array([12, 11])

    Series should be stationary before applying ACF.
    To pipeline SeasonalityACF with the Differencer, use the ParamFitterPipeline:

    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.series.difference import Differencer
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>> sp_est = Differencer() * SeasonalityACF()  # doctest: +SKIP
    >>> sp_est.fit(X)  # doctest: +SKIP
    ParamFitterPipeline(...)
    >>> sp_est.get_fitted_params()["sp"]  # doctest: +SKIP
    12
    >>> sp_est.get_fitted_params()["sp_significant"]  # doctest: +SKIP
    array([12, 11])
    """

    _tags = {
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        candidate_sp=None,
        p_threshold=0.05,
        adjusted=False,
        nlags=None,
        fft=True,
        missing="none",
    ):
        self.candidate_sp = candidate_sp
        self.p_threshold = p_threshold
        self.adjusted = adjusted
        self.nlags = nlags
        self.fft = fft
        self.missing = missing
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.stattools import acf

        p_threshold = self.p_threshold
        adjusted = self.adjusted

        nlags = self.nlags
        if nlags is None:
            nobs = len(X)
            nlags = min(10 * np.log10(nobs), nobs - 1)
            nlags = int(nlags)

        candidate_sp = self.candidate_sp
        if candidate_sp is None:
            candidate_sp = range(2, nlags + 1)
        elif isinstance(candidate_sp, int):
            candidate_sp = [candidate_sp]

        fft = self.fft
        missing = self.missing

        acf_series, confint = acf(
            x=X,
            adjusted=adjusted,
            nlags=nlags,
            fft=fft,
            missing=missing,
            alpha=p_threshold,
        )
        self.acf_ = acf_series
        self.confint_ = confint
        lower = confint[:, 0]
        reject = lower < 0

        lower_cand = lower[candidate_sp]
        reject_cand = reject[candidate_sp]

        sorting = np.argsort(-lower_cand)
        reject_ordered = reject_cand[sorting]
        sp_ordered = np.array(candidate_sp)[sorting]
        sp_significant = sp_ordered[~reject_ordered]

        if len(sp_significant) > 0:
            self.sp_ = sp_significant[0]
            self.sp_significant_ = sp_significant
        else:
            self.sp_ = 1
            self.sp_significant_ = []

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"candidate_sp": [3, 7, 12]}
        params3 = {"candidate_sp": 12}

        return [params1, params2, params3]


class SeasonalityACFqstat(BaseParamFitter):
    """Find candidate seasonality parameter using autocorrelation function LB q-stat.

    Uses `statsmodels.tsa.stattools.act` for computing the autocorrelation function,
    and uses its testing functionality to determine candidate seasonality parameters.
    ("seasonality parameter" are integer lags, and abbreviated by sp, below)

    Obtains Ljung-Box q-statistic to test for candidate sp at `candidate_sp`.

    Then applies `statsmodels.stats.multitest.multipletests` to correct multiple tests.
    Fitted attributes returned are significant sp and the most significant sp.
    These can be used in conditional or unconditional deseasonalization.

    Note: this should be applied to stationary series.
        Quick stationarity transformation can be achieved by differencing.
        See also: Differencer

    Parameters
    ----------
    candidate_sp : None, int or list of int, optional, default = None
        candidate sp to test, and to restrict tests to; ints must be 2 or larger
        if None, will test all integer lags between 2 and `nlags` (inclusive)
    p_threshold : float, optional, default=0.05
        significance threshold to apply in testing for seasonality
    p_adjust : str, optional, default="fdr_by" (Benjamini/Yekutieli)
        multiple testing correction applied to p-values of candidate sp in acf test
        multiple testing correction is applied to Ljung-Box tests on candidate_sp
        values can be "none" or strings accepted by `statsmodels` `multipletests`
        "none" = no multiple testim correction is applied, raw p-values are used
        "fdr_by" = Benjamini-Yekutieli FDR control procedure
        for other possible strings, see `statsmodels.stats.multitest.multipletests`
    adjusted : bool, optional, default=False
        If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional, default=None
        Number of lags to compute autocorrelations for and select from.
        At default None, uses `min(10 * np.log10(nobs), nobs - 1)`.
        Will be ignored if `candidate_sp` is provided.
    fft : bool, optional, default=True
        If True, computes the ACF via FFT.
    missing : str, ["none", "raise", "conservative", "drop"], optional, default="none"
        Specifies how NaNs are to be treated.
        "none" performs no checks.
        "raise" raises an exception if NaN values are found.
        "drop" removes the missing observations and treats non-missing as contiguous.
        "conservative" computes the autocovariance using nan-ops so that nans are
            removed when computing the mean and cross-products that are used to
            estimate the autocovariance. When using "conservative",
            n is set to the number of non-missing observations.

    Attributes
    ----------
    sp_ : int, seasonality period at lowest p-level, if any sub-threshold, else 1
        if `candidate_sp` is passed, will be in `candidate_sp` or 1
    sp_significant_ : list of int, seasonality periods with sub-threshold p-levels
        ordered increasingly by p-level. Empty list, not [1], if none are sub-threshold

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.seasonality import SeasonalityACFqstat
    >>> X = load_airline().diff()[1:]
    >>> sp_est = SeasonalityACFqstat(candidate_sp=[3, 7, 12])  # doctest: +SKIP
    >>> sp_est.fit(X)  # doctest: +SKIP
    SeasonalityACFqstat(...)
    >>> sp_est.get_fitted_params()["sp_significant"]  # doctest: +SKIP
    array([12,  7,  3])
    """

    _tags = {
        "X_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for X?
        "scitype:X": "Series",  # which X scitypes are supported natively?
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        candidate_sp=None,
        p_threshold=0.05,
        p_adjust="fdr_by",
        adjusted=False,
        nlags=None,
        fft=True,
        missing="none",
    ):
        self.candidate_sp = candidate_sp
        self.p_threshold = p_threshold
        self.p_adjust = p_adjust
        self.adjusted = adjusted
        self.nlags = nlags
        self.fft = fft
        self.missing = missing
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.stats.multitest import multipletests
        from statsmodels.tsa.stattools import acf

        p_threshold = self.p_threshold
        p_adjust = self.p_adjust
        adjusted = self.adjusted

        p_threshold = self.p_threshold
        adjusted = self.adjusted

        nlags = self.nlags
        if nlags is None:
            nobs = len(X)
            nlags = min(10 * np.log10(nobs), nobs - 1)
            nlags = int(nlags)

        candidate_sp = self.candidate_sp
        if candidate_sp is None:
            candidate_sp = range(2, nlags + 1)
        elif isinstance(candidate_sp, int):
            candidate_sp = [candidate_sp]

        fft = self.fft
        missing = self.missing

        acf_series, confint, qstat, pvalues = acf(
            x=X,
            adjusted=adjusted,
            nlags=nlags,
            fft=fft,
            missing=missing,
            alpha=p_threshold,
            qstat=True,
        )
        self.acf_ = acf_series
        self.confint_ = confint
        self.qstat_ = qstat
        self.pvalues_ = pvalues

        if candidate_sp is not None:
            qstat_cand = qstat[candidate_sp]
            pvalues_cand = pvalues[candidate_sp]
        else:
            qstat_cand = qstat
            pvalues_cand = pvalues
            candidate_sp = range(2, nlags + 1)

        self.qstat_cand_ = qstat_cand
        self.pvalues_cand = pvalues_cand

        if p_adjust != "none":
            reject_cand, pvals_adj, _, _ = multipletests(
                pvals=pvalues_cand, alpha=p_threshold, method=p_adjust
            )
            self.pvalues_adjusted_ = pvals_adj
        else:
            self.pvalues_adjusted_ = pvalues_cand
            reject_cand = pvalues_cand > p_threshold

        sorting = np.argsort(pvalues_cand)
        reject_ordered = reject_cand[sorting]
        sp_ordered = np.array(candidate_sp)[sorting]
        sp_significant = sp_ordered[reject_ordered]

        if len(sp_significant) > 0:
            self.sp_ = sp_significant[0]
            self.sp_significant_ = sp_significant
        else:
            self.sp_ = 1
            self.sp_significant_ = []

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"candidate_sp": [3, 7, 12]}
        params3 = {"candidate_sp": 12}

        return [params1, params2, params3]


class SeasonalityPeriodogram(BaseParamFitter):
    """Score periodicities by their spectral power.

    Interfacing `seasonal.periodogram` to determine candidate seasonality parameters.

    Parameters
    ----------
    min_period : int
        Disregard periods shorter than this number of samples.
        Defaults to 4
    max_period : int
        Disregard periods longer than this number of samples.
        Defaults to None
    thresh : float (0..1)
        Retain periods scoring above thresh*maxscore. Defaults to 0.10

    Attributes
    ----------
    sp_ : int, seasonality period with highest power, if any sub-threshold, else 1
    sp_significant_ : list of int, array of Fourier periods in descending order
        of their powers.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.seasonality import SeasonalityPeriodogram
    >>> X = load_airline().diff()[1:]  # doctest: +SKIP
    >>> sp_est = SeasonalityPeriodogram()  # doctest: +SKIP
    >>> sp_est.fit(X)  # doctest: +SKIP
    SeasonalityPeriodogram(...)
    >>> sp_est.get_fitted_params()["sp"]  # doctest: +SKIP
    6
    >>> sp_est.get_fitted_params()["sp_significant"]  # doctest: +SKIP
    array([6, 12, 14, 4, 10, 5])
    """

    _tags = {
        "X_inner_mtype": "pd.Series",
        "scitype:X": "Series",
        "capability:missing_values": True,
        "capability:multivariate": False,
        "python_dependencies": "seasonal",
    }

    def __init__(self, min_period=4, max_period=None, thresh=0.10):
        self.min_period = min_period
        self.max_period = max_period
        self.thresh = thresh
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.

        Returns
        -------
        self : reference to self
        """
        from seasonal.periodogram import periodogram_peaks

        seasons = periodogram_peaks(
            X,
            min_period=self.min_period,
            max_period=self.max_period,
            thresh=self.thresh,
        )

        seasons = [x[0] for x in seasons]

        if seasons is None or len(seasons) == 0:
            self.sp_ = 1
            self.sp_significant_ = []
        else:
            self.sp_significant_ = seasons
            self.sp_ = self.sp_significant_[0]

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"min_period": 5, "max_period": 24, "thresh": 0.1}
        params3 = {"min_period": 5}

        return [params1, params2, params3]
