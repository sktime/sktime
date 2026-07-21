# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Arps Decline Curve Analysis (DCA) forecasters.

Implements exponential, hyperbolic, and harmonic decline curve models following the
Arps (1945) methodology. Probabilistic forecasts are produced via Monte Carlo sampling
from the multivariate normal distribution over the curve-fitted parameters.
"""

__author__ = ["scuervo91"]
__all__ = ["ArpsExponential", "ArpsHyperbolic", "ArpsHarmonic"]

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sktime.forecasting.base import BaseForecaster


class _ArpsDcaBase(BaseForecaster):
    """Base class for Arps Decline Curve Analysis forecasters.

    Implements shared fitting, prediction, and probabilistic forecast logic
    for exponential, hyperbolic, and harmonic decline curve models.

    Parameters
    ----------
    qi_init : float or None
        Initial rate guess for curve fitting. If None, uses the first observed value.
    di_init : float
        Initial nominal decline rate guess for curve fitting.
    b_init : float
        Initial b-exponent guess (used by hyperbolic subclass only).
    n_samples : int
        Number of Monte Carlo samples for probabilistic forecasts.
    random_state : int, RandomState instance or None
        Seed for the random number generator used in probabilistic forecasts.
    output : str
        Whether to forecast instantaneous ``"rate"`` or ``"cumulative"`` production.
    anchor : str or False
        Anchoring strategy to align the model to the last observed value.
        ``"multiplicative"`` scales the forecast; ``"additive"`` shifts it;
        ``False`` applies no anchoring.
    base_np : float
        Cumulative production offset added when ``output="cumulative"``.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["scuervo91"],
        "maintainers": ["scuervo91"],
        "python_version": None,
        "python_dependencies": None,
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:pretrain": False,
        "capability:random_state": True,
    }

    def __init__(
        self,
        qi_init,
        di_init,
        b_init,
        n_samples,
        random_state,
        output,
        anchor,
        base_np,
        max_fit_retries,
    ):
        self.qi_init = qi_init
        self.di_init = di_init
        self.b_init = b_init
        self.n_samples = n_samples
        self.random_state = random_state
        self.output = output
        self.anchor = anchor
        self.base_np = base_np
        self.max_fit_retries = max_fit_retries
        super().__init__()

    def __post_init__(self):
        """Validate constructor parameters."""
        if self.output not in ("rate", "cumulative"):
            raise ValueError(
                f"Output must be 'rate' or 'cumulative', got {self.output!r}"
            )
        if self.anchor not in (False, "multiplicative", "additive"):
            raise ValueError(
                "anchor must be False, 'multiplicative', or 'additive', "
                f"got {self.anchor!r}"
            )

    def _get_varnames(self):
        """Return variable names from the training series."""
        return self._y.columns.tolist()

    @staticmethod
    def _index_to_float_array(index):
        """Convert a pandas index to a 1D float numpy array of days.

        - DatetimeIndex / PeriodIndex are converted to days since the Unix epoch.
          For PeriodIndex, the period start is used as the reference timestamp,
          which preserves variable period lengths (e.g., 28 vs 31 day months).
        - Integer-like indices are cast to float as-is, in their original units.
        The result is always 1D.
        """
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp(how="start")
        if isinstance(index, pd.DatetimeIndex):
            values = index.astype("int64").to_numpy() / 864e11
        else:
            values = index.astype("int64").to_numpy()
        return values.astype(float).reshape(-1)

    def _get_qi_init(self, q):
        """Return initial rate guess from data or constructor parameter."""
        if self.qi_init is not None:
            return float(self.qi_init)
        return float(max(q[0], 1e-10))

    @staticmethod
    def _estimate_di_from_data(q, t):
        """Rough nominal decline rate from endpoint ratio, if data is declining."""
        if len(q) < 2:
            return None
        t_span = float(t[-1] - t[0])
        if t_span <= 0:
            return None
        q0, q1 = float(q[0]), float(q[-1])
        if q0 <= 0 or q1 <= 0 or q1 >= q0:
            return None
        di_est = -np.log(q1 / q0) / t_span
        if not np.isfinite(di_est) or di_est <= 0:
            return None
        return di_est

    def _get_di_init(self, q, t):
        """Return initial decline rate guess from data or constructor parameter."""
        di_est = self._estimate_di_from_data(q, t)
        if di_est is not None:
            return max(di_est, 1e-10)
        return self.di_init

    def _fit_curve_params(self, t, q, p0):
        """Single curve fit attempt with given initial parameters.

        Returns (popt, pcov, converged). Does not warn on failure.
        """
        try:
            popt, pcov = curve_fit(
                self._rate_func,
                t,
                q,
                p0=p0,
                bounds=self._get_bounds(),
                maxfev=50000,
            )
            return popt, pcov, True
        except (RuntimeError, ValueError):
            n_params = len(p0)
            return p0, np.full((n_params, n_params), np.inf), False

    def _fit(self, y, X, fh):
        """Fit the decline curve model to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Univariate time series to fit. Single-column DataFrame.
        X : pd.DataFrame or None
            Exogenous data, ignored.
        fh : ForecastingHorizon or None
            Forecasting horizon, not required at fit time.

        Returns
        -------
        self : reference to self
        """
        self._pred_int_available_ = True

        t_all = self._index_to_float_array(y.index)
        self.t0_ = float(t_all[0])
        t = t_all - self.t0_
        q = y.iloc[:, 0].values.astype(float)

        p0 = np.asarray(self._get_p0(q, t), dtype=float)
        popt, pcov, converged = self._fit_curve_params(t, q, p0)

        if not converged and self.max_fit_retries > 0:
            rng = np.random.default_rng(self.random_state)
            for _ in range(self.max_fit_retries):
                perturb = np.exp(rng.uniform(-np.log(2), np.log(2), size=len(p0)))
                popt, pcov, converged = self._fit_curve_params(t, q, p0 * perturb)
                if converged:
                    break

        if not converged:
            warnings.warn(
                f"{type(self).__name__}: curve_fit did not converge after "
                f"{self.max_fit_retries + 1} attempt(s); using initial parameter "
                "guesses. Probabilistic forecasts will return naive intervals.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._pred_int_available_ = False

        self.params_ = popt
        self.params_cov_ = pcov

        t_last = t[-1]
        q_last_obs = q[-1]
        q_last_model = self._rate_func(t_last, *popt)

        if self.anchor == "multiplicative":
            self.anchor_scale_ = q_last_obs / q_last_model if q_last_model != 0 else 1.0
            self.anchor_shift_ = 0.0

        elif self.anchor == "additive":
            self.anchor_scale_ = 1.0
            self.anchor_shift_ = q_last_obs - q_last_model
        else:
            self.anchor_scale_ = 1.0
            self.anchor_shift_ = 0.0

        if converged and np.any(np.isinf(pcov)):
            warnings.warn(
                f"{type(self).__name__}: parameter covariance matrix contains "
                "infinite values. Probabilistic forecasts will return naive intervals. "
                "Try providing better initial guesses via qi_init and di_init.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._pred_int_available_ = False

        return self

    def _apply_forecast_transforms(self, y_pred):
        """Apply anchoring and cumulative offset to raw model predictions."""
        y_pred = y_pred * self.anchor_scale_ + self.anchor_shift_
        if self.output == "cumulative":
            baseline = y_pred[0] if y_pred.ndim == 1 else y_pred[:, [0]]
            y_pred = (y_pred - baseline) + self.base_np
        return y_pred

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data, ignored.

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts indexed by absolute forecast horizon.
        """
        fh_abs = fh.to_absolute_index(self.cutoff)
        t = self._index_to_float_array(fh_abs) - self.t0_
        func = self._rate_func if self.output == "rate" else self._cum_func
        y_pred = self._apply_forecast_transforms(func(t, *self.params_))

        cols = self._get_varnames()
        y_arr = np.asarray(y_pred).reshape(-1, len(cols))
        return pd.DataFrame(y_arr, index=fh_abs, columns=cols)

    def _mc_forecast_samples(self, fh_abs, t):
        """Return array of shape (n_samples, len(t)) of MC-sampled forecasts.

        Shared by ``_predict_quantiles`` and ``_predict_proba``. Falls back to
        a single deterministic sample (the point forecast) if the curve fit
        did not converge or the covariance matrix was degenerate.
        """
        func = self._rate_func if self.output == "rate" else self._cum_func

        if not self._pred_int_available_:
            warnings.warn(
                f"{type(self).__name__}: probabilistic forecasts unavailable; "
                "returning point forecast as naive prediction intervals.",
                RuntimeWarning,
                stacklevel=3,
            )
            y_point = self._apply_forecast_transforms(func(t, *self.params_))
            return np.asarray(y_point).reshape(1, -1)

        rng = np.random.default_rng(self.random_state)

        try:
            param_samples = rng.multivariate_normal(
                self.params_, self.params_cov_, size=self.n_samples
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            raise RuntimeError(
                "Parameter covariance matrix is singular or near-singular"
            ) from e

        param_samples = self._clip_param_samples(param_samples)

        y_pred = func(
            t, *[param_samples[:, [i]] for i in range(param_samples.shape[1])]
        )
        return self._apply_forecast_transforms(y_pred)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute prediction quantiles via Monte Carlo parameter sampling.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data, ignored.
        alpha : list of float
            Quantile levels in [0, 1].

        Returns
        -------
        quantiles : pd.DataFrame
            Multi-index columns ``(variable, alpha)``; rows indexed by fh.
        """
        fh_abs = fh.to_absolute_index(self.cutoff)
        t = self._index_to_float_array(fh_abs) - self.t0_

        y_pred = self._mc_forecast_samples(fh_abs, t)
        quantile_values = np.quantile(y_pred, alpha, axis=0).T

        varnames = self._get_varnames()
        col_index = pd.MultiIndex.from_product([varnames, alpha])
        return pd.DataFrame(quantile_values, index=fh_abs, columns=col_index)

    def _predict_proba(self, fh, X, marginal=True):
        """Compute fully probabilistic forecasts via Monte Carlo parameter sampling.

        Wraps the same Monte Carlo samples used by ``_predict_quantiles`` /
        ``_predict_interval`` as an ``skpro`` ``Empirical`` distribution, so
        all three probabilistic methods stay numerically consistent with
        each other.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data, ignored.
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Empirical predictive distribution.
        """
        from skpro.distributions.empirical import Empirical

        fh_abs = fh.to_absolute_index(self.cutoff)
        t = self._index_to_float_array(fh_abs) - self.t0_

        y_pred = self._mc_forecast_samples(fh_abs, t)

        varnames = self._get_varnames()
        n_samples = y_pred.shape[0]
        spl_index = pd.MultiIndex.from_product(
            [range(n_samples), fh_abs], names=["sample", *fh_abs.names]
        )
        spl = pd.DataFrame(y_pred.reshape(-1, 1), index=spl_index, columns=varnames)

        return Empirical(spl, time_indep=marginal)

    def _predict_interval(self, fh, X, coverage):
        """Compute prediction intervals via Monte Carlo parameter sampling.

        Delegates to ``_predict_quantiles`` using
        ``alpha = 0.5 - c/2`` (lower) and ``alpha = 0.5 + c/2`` (upper)
        for each coverage level ``c``.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data, ignored.
        coverage : list of float
            Nominal coverage levels in [0, 1].

        Returns
        -------
        pred_int : pd.DataFrame
            Multi-index columns ``(variable, coverage, "lower"/"upper")``;
            rows indexed by fh.
        """
        alphas = [p for c in coverage for p in (0.5 - c / 2, 0.5 + c / 2)]
        qtiles = self._predict_quantiles(fh, X, alpha=alphas)

        varnames = self._get_varnames()
        fh_abs = fh.to_absolute_index(self.cutoff)
        col_index = pd.MultiIndex.from_product([varnames, coverage, ["lower", "upper"]])

        arr = np.empty((len(fh_abs), len(varnames) * len(coverage) * 2))
        i = 0
        for var in varnames:
            for c in coverage:
                arr[:, i] = qtiles[(var, 0.5 - c / 2)].values
                arr[:, i + 1] = qtiles[(var, 0.5 + c / 2)].values
                i += 2

        return pd.DataFrame(arr, index=fh_abs, columns=col_index)


class ArpsExponential(_ArpsDcaBase):
    r"""Arps exponential decline curve forecaster.

    Models production decline as:

    .. math:: q(t) = q_i \\cdot e^{-D_i t}

    where :math:`q_i` is the initial rate and :math:`D_i` is the nominal decline rate.
    Parameters are estimated via nonlinear least squares
    (``scipy.optimize.curve_fit``). Probabilistic forecasts are produced by
    Monte Carlo sampling from the multivariate normal distribution over the
    fitted parameters using the covariance matrix from the curve fit.

    Parameters
    ----------
    qi_init : float or None, default=None
        Initial rate guess. If None, the first observed value is used.
    di_init : float, default=0.1
        Initial nominal decline rate guess.
    n_samples : int, default=1000
        Number of Monte Carlo samples for probabilistic forecasts.
    random_state : int, RandomState instance or None, default=None
        Seed for reproducible probabilistic forecasts.
    output : str, default="rate"
        ``"rate"`` for instantaneous rate, ``"cumulative"`` for cumulative production.
    anchor : str or False, default=False
        Last-observation anchoring: ``"multiplicative"``, ``"additive"``, or ``False``.
    base_np : float, default=0.0
        Cumulative production offset added when ``output="cumulative"``.
    max_fit_retries : int, default=3
        Number of additional fitting attempts with perturbed initial parameters
        if the first attempt fails to converge. Set to 0 to disable retries.
        If all attempts fail, naive (degenerate) prediction intervals are returned
        with a warning.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.forecasting.arps_dca import ArpsExponential
    >>> t = np.arange(10)
    >>> q = 1000 * np.exp(-0.1 * t)
    >>> y = pd.Series(q, index=t)
    >>> forecaster = ArpsExponential()
    >>> forecaster.fit(y, fh=[1, 2, 3])
    ArpsExponential(...)
    """

    def __init__(
        self,
        qi_init=None,
        di_init=0.1,
        n_samples=1000,
        random_state=None,
        output="rate",
        anchor=False,
        base_np=0.0,
        max_fit_retries=3,
    ):
        super().__init__(
            qi_init=qi_init,
            di_init=di_init,
            b_init=0,
            n_samples=n_samples,
            random_state=random_state,
            output=output,
            anchor=anchor,
            base_np=base_np,
            max_fit_retries=max_fit_retries,
        )

    @staticmethod
    def _rate_func(t, qi, Di):
        return qi * np.exp(-Di * t)

    @staticmethod
    def _cum_func(t, qi, Di):
        return (qi / Di) * (1 - np.exp(-Di * t))

    def _get_p0(self, q, t):
        return [self._get_qi_init(q), self._get_di_init(q, t)]

    def _get_bounds(self):
        return ([0.0, 1e-10], [np.inf, np.inf])

    def _clip_param_samples(self, samples):
        samples = samples.copy()
        samples[:, 0] = np.clip(samples[:, 0], 0.0, None)
        samples[:, 1] = np.clip(samples[:, 1], 1e-10, None)
        return samples

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            Keys ``"qi"`` (initial rate) and ``"Di"`` (decline rate).
        """
        return {"qi": self.params_[0], "Di": self.params_[1]}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set.

        Returns
        -------
        params : list of dict
        """
        return [
            {},
            {"output": "cumulative", "di_init": 0.05, "n_samples": 50},
        ]


class ArpsHyperbolic(_ArpsDcaBase):
    r"""Arps hyperbolic decline curve forecaster.

    Models production decline as:

    .. math:: q(t) = \\frac{q_i}{(1 + b D_i t)^{1/b}}

    where :math:`q_i` is the initial rate, :math:`D_i` is the nominal decline rate,
    and :math:`b` is the hyperbolic exponent (0 < b < 2 for physical decline).
    The exponential (b → 0) and harmonic (b = 1) cases are limiting forms.
    Parameters are estimated via nonlinear least squares
    (``scipy.optimize.curve_fit``). Probabilistic forecasts are produced by
    Monte Carlo sampling from the multivariate normal distribution over the
    fitted parameters using the covariance matrix from the curve fit.

    Parameters
    ----------
    qi_init : float or None, default=None
        Initial rate guess. If None, the first observed value is used.
    di_init : float, default=0.1
        Initial nominal decline rate guess.
    b_init : float, default=0.5
        Initial b-exponent guess.
    n_samples : int, default=1000
        Number of Monte Carlo samples for probabilistic forecasts.
    random_state : int, RandomState instance or None, default=None
        Seed for reproducible probabilistic forecasts.
    output : str, default="rate"
        ``"rate"`` for instantaneous rate, ``"cumulative"`` for cumulative production.
    anchor : str or False, default=False
        Last-observation anchoring: ``"multiplicative"``, ``"additive"``, or ``False``.
    base_np : float, default=0.0
        Cumulative production offset added when ``output="cumulative"``.
    max_fit_retries : int, default=3
        Number of additional fitting attempts with perturbed initial parameters
        if the first attempt fails to converge. Set to 0 to disable retries.
        If all attempts fail, naive (degenerate) prediction intervals are returned
        with a warning.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.forecasting.arps_dca import ArpsHyperbolic
    >>> t = np.arange(10)
    >>> q = 1000 / (1 + 0.5 * 0.1 * t) ** (1 / 0.5)
    >>> y = pd.Series(q, index=t)
    >>> forecaster = ArpsHyperbolic()
    >>> forecaster.fit(y, fh=[1, 2, 3])
    ArpsHyperbolic(...)
    """

    def __init__(
        self,
        qi_init=None,
        di_init=0.1,
        b_init=0.5,
        n_samples=1000,
        random_state=None,
        output="rate",
        anchor=False,
        base_np=0.0,
        max_fit_retries=3,
    ):
        super().__init__(
            qi_init=qi_init,
            di_init=di_init,
            b_init=b_init,
            n_samples=n_samples,
            random_state=random_state,
            output=output,
            anchor=anchor,
            base_np=base_np,
            max_fit_retries=max_fit_retries,
        )

    @staticmethod
    def _rate_func(t, qi, Di, b):
        return qi / (1.0 + b * Di * t) ** (1.0 / b)

    @staticmethod
    def _cum_func(t, qi, Di, b):
        u = 1.0 + b * Di * t
        return (qi / (Di * (b - 1.0))) * (np.power(u, (b - 1.0) / b) - 1.0)

    def _get_p0(self, q, t):
        return [self._get_qi_init(q), self._get_di_init(q, t), self.b_init]

    def _get_bounds(self):
        return ([0.0, 1e-10, 1e-10], [np.inf, np.inf, np.inf])

    def _clip_param_samples(self, samples):
        samples = samples.copy()
        samples[:, 0] = np.clip(samples[:, 0], 0.0, None)
        samples[:, 1] = np.clip(samples[:, 1], 1e-10, None)
        samples[:, 2] = np.clip(samples[:, 2], 1e-10, 2)
        return samples

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            Keys ``"qi"`` (initial rate), ``"Di"`` (decline rate), ``"b"`` (exponent).
        """
        return {"qi": self.params_[0], "Di": self.params_[1], "b": self.params_[2]}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set.

        Returns
        -------
        params : list of dict
        """
        return [
            {},
            {"output": "cumulative", "di_init": 0.05, "n_samples": 50},
        ]


class ArpsHarmonic(_ArpsDcaBase):
    r"""Arps harmonic decline curve forecaster.

    Special case of hyperbolic decline with b = 1:

    .. math::

        q(t) = \\frac{q_i}{1 + D_i t}, \\quad
        N_p(t) = \\frac{q_i}{D_i} \\ln(1 + D_i t)

    Parameters ``qi`` and ``Di`` are estimated via nonlinear least squares
    (``scipy.optimize.curve_fit``). Probabilistic forecasts are produced by
    Monte Carlo sampling from the multivariate normal distribution over the
    fitted parameters using the covariance matrix from the curve fit.

    Parameters
    ----------
    qi_init : float or None, default=None
        Initial rate guess. If None, the first observed value is used.
    di_init : float, default=0.1
        Initial nominal decline rate guess.
    n_samples : int, default=1000
        Number of Monte Carlo samples for probabilistic forecasts.
    random_state : int, RandomState instance or None, default=None
        Seed for reproducible probabilistic forecasts.
    output : str, default="rate"
        ``"rate"`` for instantaneous rate, ``"cumulative"`` for cumulative production.
    anchor : str or False, default=False
        Last-observation anchoring: ``"multiplicative"``, ``"additive"``, or ``False``.
    base_np : float, default=0.0
        Cumulative production offset added when ``output="cumulative"``.
    max_fit_retries : int, default=3
        Number of additional fitting attempts with perturbed initial parameters
        if the first attempt fails to converge. Set to 0 to disable retries.
        If all attempts fail, naive (degenerate) prediction intervals are returned
        with a warning.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.forecasting.arps_dca import ArpsHarmonic
    >>> t = np.arange(10)
    >>> q = 1000 / (1 + 0.1 * t)
    >>> y = pd.Series(q, index=t)
    >>> forecaster = ArpsHarmonic()
    >>> forecaster.fit(y, fh=[1, 2, 3])
    ArpsHarmonic(...)
    """

    def __init__(
        self,
        qi_init=None,
        di_init=0.1,
        n_samples=1000,
        random_state=None,
        output="rate",
        anchor=False,
        base_np=0.0,
        max_fit_retries=3,
    ):
        super().__init__(
            qi_init=qi_init,
            di_init=di_init,
            b_init=1,
            n_samples=n_samples,
            random_state=random_state,
            output=output,
            anchor=anchor,
            base_np=base_np,
            max_fit_retries=max_fit_retries,
        )

    @staticmethod
    def _rate_func(t, qi, Di):
        return qi / (1.0 + Di * t)

    @staticmethod
    def _cum_func(t, qi, Di):
        return (qi / Di) * np.log1p(Di * t)

    def _get_p0(self, q, t):
        return [self._get_qi_init(q), self._get_di_init(q, t)]

    def _get_bounds(self):
        return ([0.0, 1e-10], [np.inf, np.inf])

    def _clip_param_samples(self, samples):
        samples = samples.copy()
        samples[:, 0] = np.clip(samples[:, 0], 0.0, None)
        samples[:, 1] = np.clip(samples[:, 1], 1e-10, None)
        return samples

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            Keys ``"qi"`` (initial rate) and ``"Di"`` (decline rate).
        """
        return {"qi": self.params_[0], "Di": self.params_[1]}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set.

        Returns
        -------
        params : list of dict
        """
        return [
            {},
            {"output": "cumulative", "di_init": 0.05, "n_samples": 50},
        ]
