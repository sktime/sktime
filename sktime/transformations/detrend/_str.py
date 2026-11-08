# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Seasonal-trend decomposition using penalized regression."""

__all__ = ["STR"]

from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from sktime.transformations.base import BaseTransformer


class STR(BaseTransformer):
    """Seasonal-Trend decomposition using Regression (STR).

    Fit a piecewise linear trend and one cyclic, piecewise bilinear surface
    per seasonal period by penalized least squares [1]_. Seasonal surfaces
    can change over time. Second derivatives in time and season, and mixed
    derivatives, are penalized. Each seasonal surface has zero mean over
    its seasonal knots at every time knot, separating it from the trend.

    This implements the non-robust, fixed-smoothing, circular-seasonality
    case of R's ``stR::STRmodel``. It does not implement ``AutoSTR`` smoothing
    selection, covariates, arbitrary seasonal topologies, or confidence intervals.
    Time is measured in observation steps, irrespective of the index frequency.
    Observations must be equally spaced and contain no missing values.

    ``fit`` stores ``trend_``, ``seasonal_``, and ``resid_``. ``transform``
    returns the series with seasonal components removed, or all components
    when ``return_components=True``. New observations are decomposed together
    with the training series without changing fitted state.

    ``inverse_transform`` adds back the fitted seasonal components in sample.
    For future indices, it repeats the last cycle of each seasonal component,
    allowing use in a ``TransformedTargetForecaster``. With components output,
    inverse transformation sums the component columns.

    Parameters
    ----------
    periods : int or sequence of int, default=12
        Distinct seasonal periods, each at least 2. At least two full cycles
        of every period must be available in ``fit``.
    lambdas : float or sequence of float, default=1.0
        Positive smoothing weights multiplying derivative penalty rows.
        A scalar applies to all penalties. A sequence has length
        ``1 + 3 * len(periods)``: trend time curvature, then time curvature,
        seasonal curvature, and mixed curvature for each period, in input
        order. The squared weights enter the least-squares objective.
    trend_knots : int or None, default=None
        Number of equally spaced trend knots, at least 3. If None, use
        ``max(16, ceil(2 * n / max(periods) + 1))``.
    seasonal_time_knots : int or None, default=None
        Number of equally spaced time knots per seasonal surface, at least 3.
        If None, use ``max(3, ceil(n / max(periods) + 1))``.
    seasonal_knots : int or sequence of int or None, default=None
        Number of equally spaced cyclic knots for each seasonal surface,
        at least 2. A scalar applies to every period. If None, use
        ``max(2, floor(p / sqrt(q)))`` for each period ``p``, where ``q``
        is the largest smaller period, or 1 if there is none.
    return_components : bool, default=False
        If True, return a DataFrame with columns ``trend``, ``resid``, and
        ``seasonal_<period>``. Otherwise return a deseasonalized series.

    Attributes
    ----------
    trend_ : pd.Series
        Fitted trend, indexed like the training series.
    seasonal_ : pd.DataFrame
        Fitted seasonal components, one column per period.
    resid_ : pd.Series
        Training series minus trend and all seasonal components.

    References
    ----------
    .. [1] Dokumentov, A. and Hyndman, R.J. (2022). STR: Seasonal-Trend
       decomposition using Regression. INFORMS Journal on Data Science,
       1(1), 50-62. https://robjhyndman.com/publications/str/

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.detrend import STR
    >>> y = load_airline()
    >>> transformer = STR(periods=12, return_components=True)
    >>> components = transformer.fit_transform(y)
    >>> list(components.columns)
    ['trend', 'resid', 'seasonal_12']

    Deseasonalize before forecasting and add seasonal cycles back afterwards:

    >>> from sktime.forecasting.trend import TrendForecaster
    >>> forecaster = STR(periods=12) * TrendForecaster()
    >>> _ = forecaster.fit(y, fh=[1, 2, 3])
    >>> predictions = forecaster.predict()
    """

    _tags = {
        "authors": ["DhruvKhanna7777"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "capability:multivariate": False,
        "capability:inverse_transform": True,
        "capability:inverse_transform:exact": False,
        "capability:categorical_in_X": False,
        "transform-returns-same-time-index": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        periods=12,
        lambdas=1.0,
        trend_knots=None,
        seasonal_time_knots=None,
        seasonal_knots=None,
        return_components=False,
    ):
        self.periods = periods
        self.lambdas = lambdas
        self.trend_knots = trend_knots
        self.seasonal_time_knots = seasonal_time_knots
        self.seasonal_knots = seasonal_knots
        self.return_components = return_components
        super().__init__()

    def _fit(self, X, y=None):
        self._validate_parameters(len(X))
        self._X = X.copy()
        self.trend_, self.seasonal_, self.resid_ = self._decompose(X)
        return self

    def _validate_parameters(self, n):
        periods = [self.periods] if isinstance(self.periods, Integral) else self.periods
        if (
            not isinstance(periods, (list, tuple, np.ndarray))
            or np.ndim(periods) != 1
            or len(periods) == 0
            or any(not _is_int(p, 2) for p in periods)
            or len(set(periods)) != len(periods)
        ):
            raise ValueError("periods must contain distinct integers of at least 2.")
        self._periods = list(periods)
        if n < 2 * max(periods):
            raise ValueError("At least two full cycles of every period are required.")
        for name in ("trend_knots", "seasonal_time_knots"):
            value = getattr(self, name)
            if value is not None and not _is_int(value, 3):
                raise ValueError(f"{name} must be an integer of at least 3 or None.")
        knots = self.seasonal_knots
        if knots is None:
            knots = [
                max(2, int(p / np.sqrt(max([q for q in periods if q < p] + [1]))))
                for p in periods
            ]
        elif isinstance(knots, Integral):
            knots = [knots] * len(periods)
        if (
            not isinstance(knots, (list, tuple, np.ndarray))
            or np.ndim(knots) != 1
            or len(knots) != len(periods)
            or any(not _is_int(k, 2) for k in knots)
        ):
            raise ValueError(
                "seasonal_knots must give integers of at least 2 per period."
            )
        self._seasonal_knots = list(knots)
        weights = self.lambdas
        if isinstance(weights, Real):
            weights = [weights] * (1 + 3 * len(periods))
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != (1 + 3 * len(periods),)
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError(
                "lambdas must give positive finite weights, 1 + 3 per period."
            )
        self._lambdas = weights

    def _decompose(self, X):
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("STR requires finite observations with no missing values.")
        n = len(X)
        t = np.arange(n, dtype=float)
        largest = max(self._periods)
        nt = self.trend_knots or max(16, int(np.ceil(2 * n / largest + 1)))
        ns = self.seasonal_time_knots or max(3, int(np.ceil(n / largest + 1)))
        trend_t = np.linspace(0, n - 1, nt)
        design = [_linear_basis(t, trend_t)]
        penalties = [self._lambdas[0] * _time_derivatives(trend_t)]
        for i, (p, k) in enumerate(zip(self._periods, self._seasonal_knots)):
            time_knots = np.linspace(0, n - 1, ns)
            basis, regs = _seasonal_surface(t, p, k, time_knots)
            design.append(basis)
            weights = self._lambdas[1 + 3 * i : 4 + 3 * i]
            penalties.append(sparse.vstack([w * r for w, r in zip(weights, regs)]))
        C = sparse.hstack(design, format="csr")
        R = sparse.block_diag(penalties, format="csr")
        # Match STRmodel's penalized least-squares normal equations.
        coef = spsolve((C.T @ C + R.T @ R).tocsc(), C.T @ values)
        if not np.isfinite(coef).all():
            raise ValueError("STR system is singular; increase the smoothing weights.")
        parts = []
        offset = 0
        for basis in design:
            width = basis.shape[1]
            parts.append(basis @ coef[offset : offset + width])
            offset += width
        trend = pd.Series(parts[0], index=X.index, name=X.name)
        seasonal = pd.DataFrame(
            dict(zip([f"seasonal_{p}" for p in self._periods], parts[1:])),
            index=X.index,
        )
        resid = (X - trend - seasonal.sum(axis=1)).rename(X.name)
        return trend, seasonal, resid

    def _transform(self, X, y=None):
        if X.equals(self._X):
            trend, seasonal, resid = self.trend_, self.seasonal_, self.resid_
        else:
            full = X.combine_first(self._X).sort_index()
            trend, seasonal, resid = self._decompose(full)
        if self.return_components:
            result = pd.concat(
                [trend.rename("trend"), resid.rename("resid"), seasonal], axis=1
            )
        else:
            result = (X - seasonal.sum(axis=1)).rename(X.name)
        return result.loc[X.index].copy()

    def _inverse_transform(self, X, y=None):
        if self.return_components:
            result = X.sum(axis=1) if isinstance(X, pd.DataFrame) else X
            return result.rename(self._X.name)
        from sktime.forecasting.base import ForecastingHorizon
        from sktime.forecasting.naive import NaiveForecaster

        seasonal = self.seasonal_.reindex(X.index).copy()
        unseen = X.index[~X.index.isin(self._X.index)]
        if len(unseen):
            for p in self._periods:
                column = f"seasonal_{p}"
                nf = NaiveForecaster(strategy="last", sp=p)
                fh = ForecastingHorizon(unseen, is_relative=False)
                seasonal.loc[unseen, column] = nf.fit(
                    self.seasonal_[column], fh=fh
                ).predict()
        return (X + seasonal.sum(axis=1)).rename(self._X.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return parameter settings for estimator tests."""
        return [
            {"periods": 2},
            {"periods": [2, 3], "return_components": True, "lambdas": 2.0},
        ]


def _is_int(value, minimum):
    """Check an integer parameter, excluding booleans."""
    return (
        isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum
    )


def _linear_basis(x, knots):
    """Sparse linear interpolation weights, including both boundary knots."""
    left = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    right_weight = (x - knots[left]) / (knots[left + 1] - knots[left])
    rows = np.repeat(np.arange(len(x)), 2)
    cols = np.column_stack([left, left + 1]).ravel()
    data = np.column_stack([1 - right_weight, right_weight]).ravel()
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(x), len(knots)))


def _time_derivatives(knots):
    """Territory-weighted second derivatives for equally spaced time knots."""
    step = knots[1] - knots[0]
    n = len(knots)
    return (
        sparse.diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            [0, 1, 2],
            shape=(n - 2, n),
            format="csr",
        )
        / step**1.5
    )


def _seasonal_surface(t, period, n_seasons, time_knots):
    """Construct a centered cyclic surface and the three STR penalties.

    Uniform knots simplify the territory weights in stR's ttRegulariser,
    ssRegulariser, and stRegulariser to Kronecker products. The final cyclic
    knot is represented as minus the sum of the others, as in stR's
    seasonalTransformer; this enforces identifiability at each time knot.
    """
    k = n_seasons
    nt = len(time_knots)
    ds = period / k
    dt = time_knots[1] - time_knots[0]
    # R orders the knots ds, 2*ds, ..., period (equivalent to zero).
    phase = ((t + 1) % period) / ds
    left = np.floor(phase).astype(int)
    fraction = phase - left
    rows = np.repeat(np.arange(len(t)), 2)
    cols = np.column_stack([(left - 1) % k, left % k]).ravel()
    data = np.column_stack([1 - fraction, fraction]).ravel()
    seasonal = sparse.csr_matrix((data, (rows, cols)), shape=(len(t), k))
    time = _linear_basis(t, time_knots)
    full = sparse.hstack(
        [seasonal.multiply(time[:, [j]]) for j in range(nt)], format="csr"
    )
    center = sparse.vstack([sparse.eye(k - 1), -np.ones((1, k - 1))], format="csr")
    reduction = sparse.kron(sparse.eye(nt), center, format="csr")
    # COO sums duplicate entries for the two-knot cycle correctly.
    row = np.repeat(np.arange(k), 3)
    col = np.column_stack(
        [(np.arange(k) - 1) % k, np.arange(k), (np.arange(k) + 1) % k]
    )
    dss = (
        sparse.csr_matrix(
            (np.tile([1.0, -2.0, 1.0], k), (row, col.ravel())), shape=(k, k)
        )
        / ds**1.5
    )
    tw = np.full(nt, np.sqrt(dt))
    tw[[0, -1]] /= np.sqrt(2)
    tt = sparse.kron(_time_derivatives(time_knots), np.sqrt(ds) * sparse.eye(k))
    ss = sparse.kron(sparse.diags(tw), dss)
    dtime = sparse.diags(
        [-np.ones(nt - 1), np.ones(nt - 1)], [0, 1], shape=(nt - 1, nt)
    )
    dseason = sparse.csr_matrix(
        (
            np.tile([-1.0, 1.0], k),
            (
                np.repeat(np.arange(k), 2),
                np.column_stack([np.arange(k), (np.arange(k) + 1) % k]).ravel(),
            ),
        ),
        shape=(k, k),
    )
    st = sparse.kron(dtime, dseason) / np.sqrt(dt * ds)
    return full @ reduction, [r @ reduction for r in (tt, ss, st)]
