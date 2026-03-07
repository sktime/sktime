# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hurdle model forecaster for intermittent demand."""

__author__ = ["sktime developers"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class HurdleForecaster(BaseForecaster):
    r"""Hurdle model for probabilistic intermittent demand forecasting.

    A two-part model that separately estimates:

    1. The probability of non-zero demand (Bernoulli part), via exponential
       smoothing of the demand occurrence indicator.
    2. The expected demand size given demand occurs, via exponential smoothing
       of non-zero demand values.

    Point forecast is :math:`\hat{y} = \hat{p} \cdot \hat{\mu}`, where
    :math:`\hat{p}` is the smoothed demand probability and :math:`\hat{\mu}`
    is the smoothed mean of non-zero demand.

    Probabilistic forecasts use a hurdle distribution: a point mass at zero
    with probability :math:`1 - \hat{p}`, and a zero-truncated Poisson or
    Negative Binomial for positive values.

    Parameters
    ----------
    alpha : float, default=0.1
        Smoothing parameter for demand occurrence probability.
    beta : float, default=0.1
        Smoothing parameter for demand size.
    distribution : str, default="poisson"
        Distribution for the non-zero demand part.
        One of ``"poisson"`` or ``"negbinom"``.

    Attributes
    ----------
    demand_prob_ : float
        Fitted demand occurrence probability.
    demand_mean_ : float
        Fitted mean of non-zero demand.
    demand_var_ : float
        Fitted variance of non-zero demand (relevant for ``"negbinom"``).

    Examples
    --------
    >>> from sktime.forecasting.hurdle_model import HurdleForecaster
    >>> from sktime.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()
    >>> forecaster = HurdleForecaster(alpha=0.2, beta=0.1, distribution="poisson")
    >>> forecaster.fit(y)
    HurdleForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    >>> y_pred_interval = forecaster.predict_interval(fh=[1, 2, 3], coverage=0.9)

    See Also
    --------
    Croston : Croston's method for intermittent demand.
    TSB : Teunter-Syntetos-Babai method for intermittent demand.

    References
    ----------
    .. [1] Syntetos, A.A., Boylan, J.E. and Croston, J.D. (2005).
       On the categorization of demand patterns.
       Journal of the Operational Research Society, 56(5), pp.495-503.

    .. [2] Mullahy, J. (1986). Specification and testing of some modified count
       data models. Journal of Econometrics, 33(3), pp.341-365.
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "requires-fh-in-fit": False,
        "capability:exogenous": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "y_inner_mtype": "pd.Series",
    }

    def __init__(self, alpha=0.1, beta=0.1, distribution="poisson"):
        self.alpha = alpha
        self.beta = beta
        self.distribution = distribution
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
        fh : int, list or np.array, optional (default=None)
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self
        """
        alpha = self.alpha
        beta = self.beta
        distribution = self.distribution

        if distribution not in ("poisson", "negbinom"):
            raise ValueError(
                f"distribution must be 'poisson' or 'negbinom', got {distribution!r}"
            )

        y_vals = y.to_numpy()
        n = len(y_vals)
        nonzero_vals = y_vals[y_vals > 0]

        p = np.zeros(n + 1)
        p[0] = np.mean(y_vals > 0) if len(nonzero_vals) > 0 else 0.5
        for t in range(n):
            p[t + 1] = alpha * float(y_vals[t] > 0) + (1 - alpha) * p[t]

        mu = np.zeros(n + 1)
        mu[0] = float(np.mean(nonzero_vals)) if len(nonzero_vals) > 0 else 1.0
        for t in range(n):
            if y_vals[t] > 0:
                mu[t + 1] = beta * y_vals[t] + (1 - beta) * mu[t]
            else:
                mu[t + 1] = mu[t]

        self.demand_prob_ = float(p[-1])
        self.demand_mean_ = float(mu[-1])

        if distribution == "negbinom" and len(nonzero_vals) > 1:
            sample_var = float(np.var(nonzero_vals, ddof=1))
            self.demand_var_ = max(sample_var, self.demand_mean_ + 1e-6)
        else:
            self.demand_var_ = self.demand_mean_

        return self

    def _predict(self, fh=None, X=None):
        """Return point forecasts (E[y] = p * mu).

        Parameters
        ----------
        fh : int, list or np.array, optional (default=None)
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        y_pred : pd.Series
        """
        forecast = self.demand_prob_ * self.demand_mean_
        index = self.fh.to_absolute_index(self.cutoff)
        return pd.Series(np.full(len(self.fh), forecast), index=index, name=self._y.name)

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute prediction intervals via the hurdle distribution.

        Parameters
        ----------
        fh : ForecastingHorizon
        X : pd.DataFrame, optional (default=None)
        coverage : list of float

        Returns
        -------
        pred_int : pd.DataFrame
        """
        if coverage is None:
            coverage = [0.9]

        index = fh.to_absolute_index(self.cutoff)
        var_names = self._get_varnames()
        var_name = var_names[0]

        cols = pd.MultiIndex.from_product(
            [var_names, coverage, ["lower", "upper"]],
            names=["variable", "coverage", "bound"],
        )
        pred_int = pd.DataFrame(index=index, columns=cols, dtype=float)

        for cov in coverage:
            alpha_lo = (1 - cov) / 2
            alpha_hi = 1 - alpha_lo
            lower = self._hurdle_quantile(alpha_lo)
            upper = self._hurdle_quantile(alpha_hi)
            pred_int[(var_name, cov, "lower")] = lower
            pred_int[(var_name, cov, "upper")] = upper

        return pred_int

    def _hurdle_quantile(self, prob):
        """Return a quantile of the hurdle distribution.

        The hurdle CDF is F(0) = 1 - p_occ, and for k >= 1:
        F(k) = (1 - p_occ) + p_occ * F_trunc(k), where F_trunc is the CDF
        of the zero-truncated count distribution.

        Parameters
        ----------
        prob : float

        Returns
        -------
        q : float
        """
        from scipy.stats import nbinom, poisson

        p_occ = self.demand_prob_
        mu = self.demand_mean_
        distribution = self.distribution

        if prob <= (1 - p_occ):
            return 0.0

        p_trunc = (prob - (1 - p_occ)) / p_occ

        if distribution == "poisson" or self.demand_var_ <= mu:
            lam = max(mu, 1e-9)
            f0 = poisson.pmf(0, mu=lam)
            target = p_trunc * (1 - f0) + f0
            k = 1
            while poisson.cdf(k, mu=lam) < target and k <= 10_000:
                k += 1
        else:
            var = self.demand_var_
            r = mu**2 / (var - mu)
            p_nb = r / (r + mu)
            f0 = nbinom.pmf(0, n=r, p=p_nb)
            target = p_trunc * (1 - f0) + f0
            k = 1
            while nbinom.cdf(k, n=r, p=p_nb) < target and k <= 10_000:
                k += 1

        return float(k)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict
        """
        return [
            {},
            {"alpha": 0.2, "beta": 0.1, "distribution": "poisson"},
            {"alpha": 0.3, "beta": 0.2, "distribution": "negbinom"},
            {"alpha": 0.5, "beta": 0.5},
        ]
