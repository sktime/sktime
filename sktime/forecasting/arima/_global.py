# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ARIMA with coefficients shared across a collection of series."""

__author__ = ["renuprasads"]
__all__ = ["GlobalARIMA"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.warnings import warn

# slope applied to the distance outside the accuracy region, see _objective
_PENALTY = 1e12

# box on the unconstrained parameters handed to the optimiser
_BOUND = 1e4

# statsmodels' exact likelihood loses accuracy as the autoregressive roots
# approach the unit circle. _log_error_growth measures that loss and the
# estimator keeps it below this value. See the class docstring.
_MAX_LOG_ERROR_GROWTH = np.log(1e8)

# projected gradient below which a solution counts as a maximum, relative to
# the size of the objective. The summed log-likelihood grows with the number
# of series, so an absolute threshold would call large collections unconverged.
_GRAD_TOL = 1e-5


class GlobalARIMA(BaseForecaster):
    """ARIMA with one set of coefficients shared across many series.

    Fits a single set of autoregressive and moving average coefficients to a
    collection of series by joint maximum likelihood, then applies them to
    individual series. Each series keeps its own level and its own innovation
    variance, so series on different scales can be pooled.

    Pooling helps when series are short and share dynamics, because every
    series contributes to the same coefficient estimate. It hurts when the
    dynamics genuinely differ, since one set of coefficients then describes
    none of them well.

    Use through two methods:

    * ``pretrain(panel)`` estimates the shared coefficients from a panel or
      hierarchical collection of series.
    * ``fit(series)`` applies them to one series and estimates only that
      series' mean (when ``d = 0``) and innovation variance. The series does
      not have to appear in the pretraining data, so a series with no history
      of its own can still be forecast.

    Called without ``pretrain``, ``fit`` is ordinary single series maximum
    likelihood ARIMA.

    Parameters
    ----------
    order : tuple of three ints, default=(1, 0, 0)
        The ``(p, d, q)`` order: autoregressive order, degree of differencing,
        moving average order. ``p + q`` must be at least 1.
    maxiter : int, default=200
        Maximum number of iterations for each optimiser run.

    Attributes
    ----------
    global_ar_params_ : np.ndarray, shape (p,)
        Shared autoregressive coefficients, set by ``pretrain``.
    global_ma_params_ : np.ndarray, shape (q,)
        Shared moving average coefficients, set by ``pretrain``.
    n_pretrain_instances_ : int
        Number of series the shared coefficients were estimated from.
    pretrain_loglike_ : float
        Summed log-likelihood at the shared estimate.
    pretrain_order_ : tuple
        The ``order`` in force when ``pretrain`` ran. ``fit`` raises if
        ``order`` has changed since, because the stored coefficients no longer
        match the model being fitted.

    Notes
    -----
    Differencing is applied to each series directly, and forecasts are
    re-integrated afterwards. statsmodels' default initialisation for
    integrated models lets the level of a series enter its likelihood, which
    would let an arbitrary level distort a shared estimate.

    The innovation variance is concentrated out of each series' likelihood
    rather than shared. This is what makes the shared estimate invariant to
    rescaling or shifting any one series.

    statsmodels' exact likelihood is computed in double precision and loses
    accuracy as the autoregressive roots approach the unit circle. The loss is
    bounded by the product of the stationary variance factors and the
    conditioning of the state recursion, and the estimator keeps that product
    below 1e8, which holds the likelihood error near machine precision. When
    an estimate sits at that limit the estimator warns, which usually means
    ``d`` is too small for the data.

    For ``d = 0`` the per-series means are estimated alongside the shared
    coefficients, so the shared estimate carries the incidental parameter bias
    described by Nickell (1981). The bias is of order ``1 / T`` in the length
    of the series and vanishes for ``d >= 1``, where differencing removes the
    means.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A. (1991). Time Series: Theory and
       Methods. Springer, 2nd edition.
    .. [2] Nickell, S. (1981). Biases in dynamic models with fixed effects.
       Econometrica, 49(6), 1417-1426.
    .. [3] Montero-Manso, P. and Hyndman, R.J. (2021). Principles and
       algorithms for forecasting groups of time series: Locality and
       globality. International Journal of Forecasting, 37(4), 1632-1653.

    Examples
    --------
    >>> from sktime.datasets import load_hierarchical_sales_toydata
    >>> from sktime.forecasting.arima import GlobalARIMA
    >>>
    >>> panel = load_hierarchical_sales_toydata()
    >>> one_series = panel.loc[("Food preparation", "Hobs")]
    >>>
    >>> forecaster = GlobalARIMA(order=(1, 1, 1))  # doctest: +SKIP
    >>> forecaster.pretrain(panel)  # doctest: +SKIP
    >>> forecaster.fit(one_series)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["renuprasads"],
        "maintainers": ["renuprasads"],
        "python_dependencies": ["statsmodels"],
        # estimator type
        # --------------
        "category": ["statistical"],
        "scitype:y": "univariate",
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pretrain": True,
        "capability:update": True,
        "pretrain:fitted_params": [
            "global_ar_params_",
            "global_ma_params_",
            "n_pretrain_instances_",
            "pretrain_loglike_",
            "pretrain_order_",
        ],
        "tests:specific": ["sktime.forecasting.tests.test_global_arima"],
    }

    def __init__(self, order=(1, 0, 0), maxiter=200):
        self.order = order
        self.maxiter = maxiter
        super().__init__()

    # ------------------------------------------------------------------
    # input checks and transformation
    # ------------------------------------------------------------------

    def _check_order(self):
        """Validate ``order`` and return it as a tuple of ints."""
        order = self.order
        if not isinstance(order, (tuple, list)) or len(order) != 3:
            raise ValueError(
                f"order must be a tuple of three integers, got {self.order!r}"
            )
        try:
            p, d, q = (int(v) for v in order)
        except (TypeError, ValueError):
            raise ValueError(
                f"order must be a tuple of three integers, got {self.order!r}"
            )
        if any(v != w for v, w in zip((p, d, q), order)):
            raise ValueError(
                f"order must be a tuple of three integers, got {self.order!r}"
            )
        if p < 0 or d < 0 or q < 0:
            raise ValueError(f"order entries must be non-negative, got {self.order!r}")
        if p + q < 1:
            raise ValueError(
                "order must have p + q at least 1, otherwise there is nothing "
                f"to estimate, got {self.order!r}"
            )
        return p, d, q

    def _check_series(self, values, d, method):
        """Raise if a series cannot be used, without transforming it."""
        if len(values) <= d:
            raise ValueError(
                f"series passed to {method} has {len(values)} observations and "
                f"cannot be differenced {d} times, at least {d + 1} are needed"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"series passed to {method} contains infinite or missing values"
            )

    def _transform(self, values, d):
        """Difference or demean a series.

        Returns the transformed series, or None when it carries no variation.
        The tolerance scales with the magnitude of the series, because
        differencing a smooth trend leaves rounding error behind rather than
        exact zeros, and that rounding error must not be read as data.
        """
        if d == 0:
            z = values - np.mean(values)
        else:
            z = np.diff(values, n=d)
        rounding = 100 * 2**d * np.finfo(float).eps * np.max(np.abs(values))
        if np.ptp(z) > rounding:
            return z
        return None

    @staticmethod
    def _series_values(y):
        """Extract a float array from a Series or single column DataFrame."""
        if isinstance(y, pd.DataFrame):
            return y.iloc[:, 0].to_numpy(dtype=float)
        return np.asarray(y, dtype=float)

    # ------------------------------------------------------------------
    # likelihood and the accuracy region
    # ------------------------------------------------------------------

    def _model(self, z, p, q):
        """Build a state space model for one transformed series."""
        import warnings as _warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return SARIMAX(
                np.asarray(z, dtype=float),
                order=(p, 0, q),
                trend="n",
                concentrate_scale=True,
                enforce_stationarity=True,
                enforce_invertibility=True,
                validate_specification=False,
            )

    @staticmethod
    def _total_loglike(u, models):
        """Sum the log-likelihood over the collection, at unconstrained ``u``."""
        u = np.asarray(u, dtype=float)
        total = 0.0
        for mod in models:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                value = mod.loglike(u, transformed=False)
            if not np.isfinite(value):
                return None
            total += float(value)
        return total

    @staticmethod
    def _log_error_growth(u, p, mod):
        """Measure how much likelihood accuracy is lost at ``u``.

        The two sources are the stationary variance of the autoregressive part,
        which grows with the unconstrained parameters, and the conditioning of
        the state recursion, which degrades as the largest inverse
        autoregressive root approaches the unit circle.
        """
        u = np.asarray(u, dtype=float)
        if not np.all(np.isfinite(u)):
            return np.inf
        growth = float(np.sum(np.log1p(u[:p] ** 2)))
        if p == 0:
            return growth
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ar = np.asarray(mod.transform_params(u)[:p], dtype=float)
        if not np.all(np.isfinite(ar)):
            return np.inf
        roots = np.roots(np.r_[1.0, -ar])
        lam = float(np.max(np.abs(roots))) if roots.size else 0.0
        lam = min(lam, 1.0 - 1e-12)
        return growth - float(np.log1p(-(lam**2)))

    def _shrink(self, u, p, mod):
        """Pull ``u`` back along its own direction until it meets the limit.

        Returns the projected vector and its distance from ``u``. The search is
        bounded: a root find with a capped iteration count, and if that lands
        outside the region, a fixed number of bisection steps.
        """
        u = np.asarray(u, dtype=float)
        norm = float(np.linalg.norm(u))
        if norm == 0.0 or not np.all(np.isfinite(u)):
            return np.zeros_like(u), norm if np.isfinite(norm) else 0.0

        def excess(t):
            return self._log_error_growth(t * u, p, mod) - _MAX_LOG_ERROR_GROWTH

        if excess(1.0) <= 0.0:
            return u, 0.0

        t = 0.0
        try:
            from scipy.optimize import brentq

            root = brentq(excess, 0.0, 1.0, xtol=1e-14, rtol=1e-15, maxiter=200)
            candidate = root * (1 - 1e-10)
            if excess(candidate) <= 0.0:
                t = candidate
        except (ValueError, RuntimeError):
            t = 0.0

        if t == 0.0:
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if excess(mid) <= 0.0:
                    lo = mid
                else:
                    hi = mid
            t = lo

        return t * u, float(np.linalg.norm(u) * (1.0 - t))

    def _objective(self, u, models, p):
        """Negative summed log-likelihood, continuous at the accuracy limit.

        Inside the region this is the negative summed log-likelihood. Outside,
        it is the value at the radial projection onto the limit plus a slope
        times the distance travelled. The two agree on the boundary, so the
        optimiser sees no step and is pushed back rather than stopped.
        """
        u = np.asarray(u, dtype=float)
        if not np.all(np.isfinite(u)):
            return _PENALTY

        if self._log_error_growth(u, p, models[0]) <= _MAX_LOG_ERROR_GROWTH:
            total = self._total_loglike(u, models)
            return _PENALTY if total is None else -total

        u_proj, dist = self._shrink(u, p, models[0])
        total = self._total_loglike(u_proj, models)
        base = _PENALTY if total is None else -total
        return base + _PENALTY * dist

    # ------------------------------------------------------------------
    # estimation
    # ------------------------------------------------------------------

    def _start_params(self, models, p, q):
        """Build the optimiser's starting point, averaged over the collection."""
        import warnings as _warnings

        from statsmodels.tools.sm_exceptions import EstimationWarning

        starts = []
        for mod in models:
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", category=EstimationWarning)
                _warnings.filterwarnings("ignore", category=RuntimeWarning)
                sp = np.asarray(mod.start_params, dtype=float)
            if sp.size >= p + q and np.all(np.isfinite(sp[: p + q])):
                starts.append(sp[: p + q])

        if not starts:
            return np.zeros(p + q)

        constrained = np.mean(np.asarray(starts, dtype=float), axis=0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            try:
                u = np.asarray(models[0].untransform_params(constrained), dtype=float)[
                    : p + q
                ]
            except (ValueError, FloatingPointError):
                u = np.zeros(p + q)
        if not np.all(np.isfinite(u)):
            u = np.zeros(p + q)

        u = np.clip(u, -_BOUND, _BOUND)
        u, _ = self._shrink(u, p, models[0])
        return u

    def _converged(self, res):
        """Judge convergence by the projected gradient, not the exit status.

        L-BFGS-B reports a non-zero status at points that are maxima, and
        reports success at points that are not, so the gradient is checked
        directly. Components held at a bound by the gradient are dropped,
        which is what makes this the right test on a box constrained problem.
        """
        x = np.asarray(res.x, dtype=float)
        jac = getattr(res, "jac", None)
        if jac is None:
            return bool(getattr(res, "status", 1) == 0)
        g = np.asarray(jac, dtype=float)
        if g.shape != x.shape or not np.all(np.isfinite(g)):
            return False
        proj = g.copy()
        proj[(x <= -_BOUND + 1e-12) & (g > 0)] = 0.0
        proj[(x >= _BOUND - 1e-12) & (g < 0)] = 0.0
        scale = max(1.0, abs(float(res.fun)))
        return bool(np.linalg.norm(proj) <= _GRAD_TOL * scale)

    def _minimize(self, start, models, p):
        """One bounded quasi-Newton run from ``start``."""
        from scipy.optimize import minimize

        bounds = [(-_BOUND, _BOUND)] * len(start)
        res = minimize(
            self._objective,
            np.asarray(start, dtype=float),
            args=(models, p),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "maxfun": 50 * self.maxiter},
        )
        return res, self._converged(res)

    def _search(self, models, p, q):
        """Maximise the summed likelihood, with fallbacks if the first run stops.

        The first run is a bounded quasi-Newton search from a data driven
        start. If its projected gradient says it stopped short of a maximum, a
        derivative free search and two further quasi-Newton runs follow, and
        the best point found is kept.
        """
        from scipy.optimize import minimize

        start = self._start_params(models, p, q)
        best, converged = self._minimize(start, models, p)
        if converged:
            return best, True

        simplex = minimize(
            self._objective,
            np.asarray(start, dtype=float),
            args=(models, p),
            method="Nelder-Mead",
            options={
                "maxiter": 200 * (p + q + 1),
                "fatol": 1e-10,
                "xatol": 1e-10,
            },
        )
        for candidate in (simplex.x, np.zeros(p + q)):
            trial, trial_converged = self._minimize(candidate, models, p)
            if trial.fun < best.fun:
                best, converged = trial, trial_converged

        return best, converged

    def _estimate(self, z_list, p, q, method):
        """Estimate shared coefficients from one or more transformed series."""
        models = [self._model(z, p, q) for z in z_list]
        res, converged = self._search(models, p, q)

        u = np.clip(np.asarray(res.x, dtype=float), -_BOUND, _BOUND)
        u, _ = self._shrink(u, p, models[0])

        if not converged:
            warn(
                f"{type(self).__name__}.{method}: the likelihood search did not "
                "reach a maximum. The result is the best point found. Consider "
                "a lower order or a longer series.",
                obj=self,
            )

        growth = self._log_error_growth(u, p, models[0])
        if growth >= _MAX_LOG_ERROR_GROWTH - 1e-8:
            warn(
                f"{type(self).__name__}.{method}: the estimate sits at the limit "
                "of where the likelihood can be computed accurately, which "
                "usually means the order of differencing d is too low for this "
                "data. Treat the coefficients and any forecast from them with "
                "caution.",
                obj=self,
            )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            params = np.asarray(models[0].transform_params(u), dtype=float)[: p + q]

        total = self._total_loglike(u, models)
        loglike = float(total) if total is not None else float("nan")
        return params, loglike

    # ------------------------------------------------------------------
    # fitted state and forecasts
    # ------------------------------------------------------------------

    def _set_state(self, values):
        """Filter one series with the coefficients in force."""
        p, d, q = self._order
        z = self._transform(values, d)

        if z is None or self._params is None:
            self._results = None
            if self._params is None:
                self._z_const = 0.0 if d == 0 else float(np.diff(values, n=d)[0])
            return

        mod = self._model(z, p, q)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self._results = mod.filter(np.asarray(self._params, dtype=float))

    def _integrate(self, z_forecast, values, d):
        """Undo ``d`` differences, anchoring on the observed series."""
        out = np.asarray(z_forecast, dtype=float)
        for level in range(d, 0, -1):
            anchor = float(np.diff(values, n=level - 1)[-1])
            out = anchor + np.cumsum(out)
        return out

    # ------------------------------------------------------------------
    # sktime interface
    # ------------------------------------------------------------------

    def _pretrain(self, y, X=None, fh=None):
        """Estimate shared coefficients from a collection of series."""
        p, d, q = self._check_order()

        if isinstance(y, pd.Series):
            y = y.to_frame()

        if isinstance(y.index, pd.MultiIndex):
            # every level but the last identifies a series; a one element list
            # would group by tuples in future pandas versions
            levels = list(range(y.index.nlevels - 1))
            groups = y.groupby(level=levels[0] if len(levels) == 1 else levels)
            n_instances = groups.ngroups
            frames = [frame for _, frame in groups]
        else:
            n_instances = 1
            frames = [y]

        z_list = []
        for frame in frames:
            for column in frame.columns:
                values = frame[column].to_numpy(dtype=float)
                self._check_series(values, d, "pretrain")
                z = self._transform(values, d)
                if z is not None:
                    z_list.append(z)

        if not z_list:
            raise ValueError(
                "none of the series passed to pretrain carry any variation "
                "after differencing, so there is nothing to estimate"
            )

        params, loglike = self._estimate(z_list, p, q, "pretrain")

        self.global_ar_params_ = params[:p]
        self.global_ma_params_ = params[p : p + q]
        self.n_pretrain_instances_ = int(n_instances)
        self.pretrain_loglike_ = loglike
        self.pretrain_order_ = (p, d, q)
        return self

    def _fit(self, y, X=None, fh=None):
        """Apply coefficients to one series, or estimate them from it."""
        p, d, q = self._check_order()
        self._order = (p, d, q)

        values = self._series_values(y)
        self._check_series(values, d, "fit")

        self._uses_pretrained = hasattr(self, "global_ar_params_")

        if self._uses_pretrained:
            if tuple(self.pretrain_order_) != (p, d, q):
                raise ValueError(
                    f"order was {tuple(self.pretrain_order_)} when pretrain ran "
                    f"and is {(p, d, q)} now. The shared coefficients do not "
                    "match this model. Call pretrain again, or restore the "
                    "original order."
                )
            self._params = np.r_[
                np.asarray(self.global_ar_params_, dtype=float),
                np.asarray(self.global_ma_params_, dtype=float),
            ]
        else:
            z = self._transform(values, d)
            if z is None:
                self._params = None
            else:
                self._params, _ = self._estimate([z], p, q, "fit")

        self._mean = float(np.mean(values)) if d == 0 else 0.0
        self._set_state(values)
        self._sigma2 = float(self._results.scale) if self._results is not None else 0.0
        self._cur_y = y
        return self

    def _update(self, y, X=None, update_params=True):
        """Extend the series, and re-estimate if asked."""
        from sktime.datatypes import update_data

        p, d, q = self._order
        self._cur_y = update_data(self._cur_y, y)
        values = self._series_values(self._cur_y)
        self._check_series(values, d, "update")

        if update_params:
            if not self._uses_pretrained:
                z = self._transform(values, d)
                if z is None:
                    self._params = None
                else:
                    self._params, _ = self._estimate([z], p, q, "update")
            if d == 0:
                self._mean = float(np.mean(values))

        self._set_state(values)

        if update_params:
            self._sigma2 = (
                float(self._results.scale) if self._results is not None else 0.0
            )
        return self

    def _predict(self, fh, X=None):
        """Forecast the series passed to ``fit``."""
        p, d, q = self._order
        values = self._series_values(self._cur_y)

        steps = np.asarray(fh.to_relative(self.cutoff), dtype=int)
        horizon = int(np.max(steps))

        if self._results is None:
            z_forecast = np.full(horizon, getattr(self, "_z_const", 0.0))
        else:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                z_forecast = np.asarray(
                    self._results.forecast(steps=horizon), dtype=float
                )

        if d == 0:
            y_forecast = self._mean + z_forecast
        else:
            y_forecast = self._integrate(z_forecast, values, d)

        index = fh.to_absolute_index(self.cutoff)
        picked = y_forecast[steps - 1]

        if isinstance(self._cur_y, pd.DataFrame):
            return pd.DataFrame(picked, index=index, columns=list(self._cur_y.columns))
        return pd.Series(picked, index=index, name=self._cur_y.name)

    def _get_fitted_params(self):
        """Report the state of the fitted forecaster."""
        p, d, q = self._order
        params = self._params
        return {
            "ar_params": (
                np.asarray(params[:p], dtype=float)
                if params is not None
                else np.zeros(p)
            ),
            "ma_params": (
                np.asarray(params[p : p + q], dtype=float)
                if params is not None
                else np.zeros(q)
            ),
            "mean": self._mean,
            "sigma2": self._sigma2,
            "uses_pretrained": self._uses_pretrained,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        return [{"order": (1, 0, 0)}, {"order": (1, 1, 1), "maxiter": 50}]
