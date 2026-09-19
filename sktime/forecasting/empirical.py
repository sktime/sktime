# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Empirical distribution forecaster.

Forecasts by treating the values in a fixed-size window ending at the cutoff
as an empirical distribution and projecting it, unchanged, across the entire
forecast horizon.
"""

__author__ = ["OfficialAbhinavSingh"]
__all__ = ["EmpiricalDistributionForecaster"]

import numpy as np
import pandas as pd

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster


class EmpiricalDistributionForecaster(BaseForecaster):
    """Forecast using an empirical distribution from a trailing window.

    Collects the last ``window_length`` observations up to and including the
    cutoff and uses them as a sample from the predictive distribution.  The
    distribution
    is the *same* for every step in the forecast horizon (constant-across-
    the-horizon design).

    Point predictions (``predict``) return the sample mean of the window.
    Probabilistic predictions (``predict_proba``) return an ``skpro``
    ``Empirical`` distribution constructed from the window values, replicated
    identically at every forecast time-step.

    Parameters
    ----------
    window_length : int, default=10
        Number of most-recent observations (up to and including the cutoff)
        to use as the
        empirical distribution sample.  Must be a positive integer (>= 1).
        If ``window_length`` exceeds the length of the training series, the
        entire series is used.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.empirical import EmpiricalDistributionForecaster
    >>> y = load_airline()
    >>> forecaster = EmpiricalDistributionForecaster(window_length=12)
    >>> forecaster.fit(y)
    EmpiricalDistributionForecaster(window_length=12)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["OfficialAbhinavSingh"],
        "maintainers": ["OfficialAbhinavSingh"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
    }

    def __init__(self, window_length=10):
        self.window_length = window_length

        super().__init__()

    def __post_init__(self):
        """Validate constructor parameters."""
        wl = self.window_length
        if not isinstance(wl, (int, np.integer)) or wl < 1:
            raise ValueError(
                f"window_length must be a positive integer (>= 1), got {wl!r}."
            )

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Stores the last ``window_length`` values of ``y`` as the empirical
        sample used for both point and probabilistic forecasts.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame or None
            Exogenous data — ignored.
        fh : ForecastingHorizon or None
            Forecasting horizon — not required at fit time.

        Returns
        -------
        self : reference to self.
        """
        wl = self.window_length
        self.window_values_ = np.asarray(y.iloc[-wl:].values, dtype=float)
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Returns the mean of the stored empirical window at every requested
        forecast step.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame or None
            Exogenous time series — ignored.

        Returns
        -------
        y_pred : pd.Series
            Point predictions (sample mean, constant across the horizon).
        """
        fh_abs = fh.to_absolute_index(self.cutoff)
        mean_val = np.mean(self.window_values_)
        return pd.Series(mean_val, index=fh_abs, name=self._y.name)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute prediction quantiles from the stored empirical window.

        Quantiles are the sample quantiles of the stored window, and are
        identical at every forecast time-step.  Implemented natively in
        ``numpy`` so that ``predict_quantiles``, ``predict_interval`` and
        ``predict_var`` remain available when the optional dependency
        ``skpro`` is not installed.

        The quantile convention mirrors that of the ``skpro`` ``Empirical``
        distribution returned by ``_predict_proba``, so that the two agree
        exactly whether or not ``skpro`` is installed.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data — ignored.
        alpha : list of float
            Quantile levels in [0, 1].

        Returns
        -------
        quantiles : pd.DataFrame
            Multi-index columns ``(variable, alpha)``; rows indexed by fh.
        """
        fh_abs = fh.to_absolute_index(self.cutoff)

        # empirical quantile, equal weights - same convention as skpro's
        # ``Empirical``: smallest sample value whose cumulative weight
        # reaches the quantile level
        window_sorted = np.sort(self.window_values_)
        n_samples = len(window_sorted)
        cum_weights = np.cumsum(np.full(n_samples, 1 / n_samples))
        ix = np.searchsorted(cum_weights, alpha)
        qvals = window_sorted[np.clip(ix, 0, n_samples - 1)]

        # the distribution is constant across the horizon, so tile the
        # quantiles down the fh axis
        arr = np.tile(qvals, (len(fh_abs), 1))

        varnames = self._get_varnames()
        col_index = pd.MultiIndex.from_product([varnames, alpha])
        return pd.DataFrame(arr, index=fh_abs, columns=col_index)

    def _predict_var(self, fh, X=None, cov=False):
        """Compute variance forecasts from the stored empirical window.

        The variance is the sample variance of the stored window, and is
        identical at every forecast time-step.  Implemented natively in
        ``numpy`` so that ``predict_var`` remains available when the optional
        dependency ``skpro`` is not installed.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data — ignored.
        cov : bool, optional (default=False)
            Whether to return the covariance matrix.  The predictive
            distribution is identical and independent across the horizon, so
            marginal variances are returned in either case.

        Returns
        -------
        pred_var : pd.DataFrame
            Variance forecasts, rows indexed by fh, columns the variable names.
        """
        fh_abs = fh.to_absolute_index(self.cutoff)
        var_val = np.var(self.window_values_)

        varnames = self._get_varnames()
        return pd.DataFrame(var_val, index=fh_abs, columns=varnames)

    def _predict_proba(self, fh, X, marginal=True):
        """Compute fully probabilistic forecasts.

        Wraps the stored window values as an ``skpro`` ``Empirical``
        distribution, replicated identically at every forecast time-step.

        If ``skpro`` is not installed, falls back to the default
        implementation in ``BaseForecaster``, which derives a distribution
        from ``_predict_quantiles``.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data — ignored.
        marginal : bool, optional (default=True)
            Whether returned distribution is marginal by time index.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Empirical predictive distribution.
        """
        if not _check_soft_dependencies("skpro", severity="none"):
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        from skpro.distributions.empirical import Empirical

        fh_abs = fh.to_absolute_index(self.cutoff)
        window = self.window_values_
        n_samples = len(window)

        # tile the window across every forecast time-step
        # shape: (n_samples, len(fh)) → flatten to (n_samples * len(fh),)
        y_tiled = np.tile(window[:, np.newaxis], (1, len(fh_abs)))

        spl_index = pd.MultiIndex.from_product(
            [range(n_samples), fh_abs], names=["sample", *fh_abs.names]
        )

        varnames = self._get_varnames()
        spl = pd.DataFrame(y_tiled.reshape(-1, 1), index=spl_index, columns=varnames)

        return Empirical(spl, time_indep=marginal)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or ``MyClass(**params[i])``
            creates a valid test instance.
            ``create_test_instance`` uses the first (or only) dictionary in
            ``params``.
        """
        params1 = {"window_length": 3}
        params2 = {"window_length": 7}
        return [params1, params2]
