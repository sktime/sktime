"""
Implements residual boosting chain forecaster.

Combines a base forecaster with one or more residual forecasters in sequence,
where each stage models the residuals of the previous stage's predictions.
"""

# copyright: sktime developers, BSD-3-Clause

__all__ = ["ResidualBoostingChainForecaster"]
__author__ = ["Sanchay117", "felipeangelimvieira"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class ResidualBoostingChainForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """Residual boosting forecast fitting one forecaster on residuals of another.

    Residual boosting can be used for:

    * improving forecasts from one forecaster with another, by using either
      as ``base_forecaster`` or ``residual_forecaster``
    * adding exogenous capability to a forecaster, by using it as
      ``residual_forecaster``, and fitting it
      on the residuals of an exogenous capable ``base_forecaster``
    * adding probabilistic forecasting capability to a forecaster,
      by using it as ``base_forecaster``,
      and adding probability forecasts from a probabilistic forecaster
      used as ``residual_forecaster``

    In ``fit``: fits ``base_forecaster`` to ``y`` and ``X``,
    computes in-sample residuals, and fits ``residual_forecaster``
    to the residuals and ``X``.

    In ``predict``, it predicts with both ``base_forecaster``
    and ``residual_forecaster``, and returns the sum of the two.

    Probabilistic forecasts are obtained by shifting quantiles of the residuals
    forecast by ``residual_forecaster`` by the point forecast of the
    ``base_forecaster``. This requires ``residual_forecaster`` to support
    probabilistic forecasts, but not ``base_forecaster``.

    Parameters
    ----------
    base_forecaster : sktime forecaster
        Point-forecast model that may ignore X.
    residual_forecaster : sktime forecaster or list of sktime forecasters
        Model(s) trained on the base model's in-sample residuals.
        If a list, the forecasters are applied sequentially to the residuals
        of the previous stage.

    Example
    -------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.residual_booster import (
    ...     ResidualBoostingChainForecaster,
    ... )
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sklearn.linear_model import LinearRegression
    >>> y, X = load_longley()
    >>> fh = [1, 2, 3]
    >>> base = NaiveForecaster(strategy="last")
    >>> resid = make_reduction(
    ...     estimator=LinearRegression(),
    ...     strategy="recursive",
    ...     window_length=3,
    ... )
    >>> booster = ResidualBoostingChainForecaster([base, resid]).fit(y, X=X, fh=fh)
    >>> y_pred = booster.predict(fh, X=X)
    """

    _tags = {
        "authors": ["Sanchay117", "felipeangelimvieira"],
        "capability:pred_int": True,
        "capability:exogenous": False,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "requires-fh-in-fit": False,
        "capability:categorical_in_X": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    def __init__(self, forecasters):
        self.forecasters = forecasters
        super().__init__()

        if not isinstance(forecasters, (list, tuple)) or len(forecasters) < 2:
            raise ValueError(
                "forecasters must be a list/tuple of length >= 2: "
                "[base, resid1, resid2, ...] or a list of (name, estimator) tuples."
            )

        # allow either [est, est, ...] or [(name, est), ...]
        if all(isinstance(x, tuple) and len(x) == 2 for x in forecasters):
            user_names, ests = zip(*forecasters)
            user_names = list(user_names)
            ests = list(ests)
        else:
            user_names = [None] * len(forecasters)
            ests = list(forecasters)

        checked = self._check_estimators(
            ests,
            attr_name="forecasters",
            cls_type=BaseForecaster,
            allow_mix=True,
            allow_empty=False,
            clone_ests=False,
        )

        auto_names = [nm for nm, _ in checked]
        ests = [est for _, est in checked]

        # prefer user-provided names if present, else auto names
        names = [
            n if isinstance(n, str) and n else auto_names[i]
            for i, n in enumerate(user_names)
        ]
        names = self._get_estimator_names(list(zip(names, ests)), make_unique=True)
        self._check_names(names)

        self._steps = list(zip(names, ests))

        children = [est for _, est in self._steps]
        residuals = children[1:]

        exog = all(est.get_tag("capability:exogenous") for est in children)
        miss = all(est.get_tag("capability:missing_values") for est in children)
        pred_int = all(est.get_tag("capability:pred_int") for est in children)
        in_sample = any(est.get_tag("capability:insample") for est in children)
        cat = all(est.get_tag("capability:categorical_in_X") for est in children)
        pred_int_insample = bool(residuals) and all(
            est.get_tag("capability:pred_int:insample") for est in residuals
        )

        self.set_tags(
            **{
                "capability:exogenous": exog,
                "capability:missing_values": miss,
                "capability:insample": in_sample,
                "capability:pred_int:insample": pred_int_insample,
                "capability:pred_int": pred_int,
                "capability:categorical_in_X": cat,
            }
        )

    def _fit(self, y, X=None, fh=None):
        """
        Fit base forecaster and (optionally multiple) residual forecasters.

        1) Fit clone A of base_forecaster on (y, X) to get ŷ_base(insample),
        then compute residual target r0 = y - ŷ_base(insample).
        2) Fit clone B of base_forecaster on (y, X) with final fh.
        3) If exactly one residual forecaster: fit it on r0 (no in-sample predict).
        If multiple residual forecasters: sequentially fit each on the current
        residual target and update it using each stage's in-sample prediction.
        4) Expose fitted children via `steps_`.
        """
        if isinstance(y.index, pd.MultiIndex):
            time_idx = y.index.get_level_values(-1).unique()
        else:
            time_idx = y.index
        insample_fh = ForecastingHorizon(time_idx, is_relative=False)

        base = self._steps[0][1]
        residual_steps = self._steps[1:]

        # 1) base (insample) to get residual target r0
        self.base_insample_ = clone(base).fit(y, X, fh)
        y_base_ins = self.base_insample_.predict(fh=insample_fh, X=X)
        resid_target = y - y_base_ins  # r0

        # 2) base (future) aware of final fh
        self.base_future_ = clone(base).fit(y, X, fh)

        # 3) residual stages
        self._resid_futures_ = []

        if len(residual_steps) == 1:
            # single residual: fit directly on r0, no in-sample prediction required
            name, est = residual_steps[0]
            est_future = clone(est).fit(resid_target, X, fh)
            self._resid_futures_.append((name, est_future))
        else:
            # multi-stage: require in-sample predict capability
            for name, est in residual_steps:
                if not est.get_tag("capability:insample"):
                    raise NotImplementedError(
                        f"Residual forecaster '{name}' does not support in-sample "
                        "prediction, which is required for "
                        "multi-stage residual boosting."
                    )

            r = resid_target
            for name, est in residual_steps:
                est_ins = clone(est).fit(r, X, fh)

                rhat_ins = est_ins.predict(fh=insample_fh, X=X)

                # Store a fresh clone trained on the same target r for future prediction
                est_future = clone(est).fit(r, X, fh)
                self._resid_futures_.append((name, est_future))

                # Update residual target for next stage
                r = r - rhat_ins

        # 4) expose fitted children
        self.steps_ = [("base", self.base_future_), *self._resid_futures_]
        return self

    def _predict(self, fh=None, X=None):
        """
        Forecast = base forecast + residual forecast.

        1. Use clone B of base_forecaster to obtain a prediction y_pred_base
        2. Use residual_forecaster clone to obtain a prediction y_pred_resid
        3. Return y_pred_base + y_pred_resid
        """
        y_base = self.base_future_.predict(fh=fh, X=X)
        idx = y_base.index

        y_hat = y_base
        for _, f in getattr(self, "_resid_futures_", []):
            y_add = f.predict(fh=fh, X=X)
            y_add = y_add.reindex(idx).fillna(0)
            y_hat = y_hat + y_add

        return y_hat

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Intervals from the last residual stage, shifted by earlier components."""
        resid_stages = getattr(self, "_resid_futures_", [])

        if not resid_stages:
            return self.base_future_.predict_interval(fh=fh, X=X, coverage=coverage)

        y_shift = self.base_future_.predict(fh=fh, X=X)
        if len(resid_stages) > 1:
            for _, f in resid_stages[:-1]:
                y_add = f.predict(fh=fh, X=X)
                if hasattr(y_add, "reindex"):
                    y_add = y_add.reindex(y_shift.index)
                y_add = y_add.fillna(0)
                y_shift = y_shift + y_add

        _, f_last = resid_stages[-1]
        I_last = f_last.predict_interval(fh=fh, X=X, coverage=coverage)

        y_shift_aligned = y_shift
        if hasattr(I_last, "index") and hasattr(y_shift, "reindex"):
            if not I_last.index.equals(y_shift.index):
                y_shift_aligned = y_shift.reindex(I_last.index)

        return I_last.add(y_shift_aligned, axis=1, level=0)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Quantiles from the last residual stage, shifted by earlier components."""
        resid_stages = getattr(self, "_resid_futures_", [])

        if not resid_stages:
            return self.base_future_.predict_quantiles(fh=fh, X=X, alpha=alpha)

        y_shift = self.base_future_.predict(fh=fh, X=X)
        if len(resid_stages) > 1:
            for _, f in resid_stages[:-1]:
                y_add = f.predict(fh=fh, X=X)
                if hasattr(y_add, "reindex"):
                    y_add = y_add.reindex(y_shift.index)
                y_add = y_add.fillna(0)
                y_shift = y_shift + y_add

        _, f_last = resid_stages[-1]
        Q_last = f_last.predict_quantiles(fh=fh, X=X, alpha=alpha)

        y_shift_aligned = y_shift
        if hasattr(Q_last, "index") and hasattr(y_shift, "reindex"):
            if not Q_last.index.equals(y_shift.index):
                y_shift_aligned = y_shift.reindex(Q_last.index)

        return Q_last.add(y_shift_aligned, axis=1, level=0)

    def _predict_var(self, fh, X=None, cov=False):
        """Variance/covariance from the last residual stage."""
        resid_stages = getattr(self, "_resid_futures_", [])
        if not resid_stages:
            return self.base_future_.predict_var(fh=fh, X=X, cov=cov)

        _, f_last = resid_stages[-1]
        V_last = f_last.predict_var(fh=fh, X=X, cov=cov)
        return V_last

    def _predict_proba(self, fh, X=None, marginal=True):
        """Combine full distribution forecasts from base & residual models."""
        if not _check_soft_dependencies("skpro", severity="none"):
            from sktime.utils.warnings import warn

            warn(
                "ResidualBoostingForecaster.predict_proba: optional "
                "dependency 'skpro' not found. "
                "Falling back to the default normal approximation via BaseForecaster. "
                "Install 'skpro' to enable exact shifted-distribution composition.",
                category=UserWarning,
                obj=self,
            )
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        from skpro.distributions import MeanScale

        resid_stages = getattr(self, "_resid_futures_", [])
        if not resid_stages:
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        y_shift = self.base_future_.predict(fh=fh, X=X)
        if len(resid_stages) > 1:
            for _, f in resid_stages[:-1]:
                y_add = f.predict(fh=fh, X=X)
                if hasattr(y_add, "reindex"):
                    y_add = y_add.reindex(y_shift.index)
                y_add = y_add.fillna(0)
                y_shift += y_add

        _, f_last = resid_stages[-1]
        if not hasattr(f_last, "predict_proba"):
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        p_last = f_last.predict_proba(fh=fh, X=X, marginal=marginal)

        mu = y_shift
        if hasattr(p_last, "index") and hasattr(y_shift, "reindex"):
            if not y_shift.index.equals(p_last.index):
                mu = y_shift.reindex(p_last.index)

        return MeanScale(d=p_last, mu=mu, sigma=1)

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
        params : dict or list of dict
            Parameters to create test instances of the estimator.
        """
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import YfromX
        from sktime.forecasting.naive import NaiveForecaster

        params1 = {
            "forecasters": [
                NaiveForecaster(strategy="last"),
                NaiveForecaster(strategy="mean"),
            ]
        }

        params2 = {
            "forecasters": [
                YfromX(
                    estimator=LinearRegression(),
                    pooling="local",
                ),
                NaiveForecaster(strategy="mean"),
            ]
        }

        params3 = {
            "forecasters": [
                NaiveForecaster(strategy="last"),
                NaiveForecaster(strategy="last", sp=7),
                NaiveForecaster(strategy="last", sp=12),
            ],
        }

        return [params1, params2, params3]
