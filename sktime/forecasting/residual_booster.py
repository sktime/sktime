"""
Implements a residual boosting forecaster.

Which is an easy way to turn a forecaster without exogenous capability into one with.
"""

# copyright: sktime developers, BSD-3-Clause

__all__ = ["ResidualBoostingForecaster"]
__author__ = ["Sanchay117", "felipeangelimvieira"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class ResidualBoostingForecaster(_HeterogenousMetaEstimator, BaseForecaster):
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
    >>> from sktime.forecasting.residual_booster import ResidualBoostingForecaster
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
    >>> booster = ResidualBoostingForecaster(base, resid).fit(y, X=X, fh=fh)
    >>> y_pred = booster.predict(fh, X=X)
    """

    _tags = {
        "authors": ["Sanchay117", "felipeangelimvieira"],
        "capability:pred_int": True,
        "capability:exogenous": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "requires-fh-in-fit": False,
        "capability:categorical_in_X": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    def __init__(self, base_forecaster, residual_forecaster):
        self.base_forecaster = base_forecaster
        self.residual_forecaster = residual_forecaster
        super().__init__()

        exog = self.base_forecaster.get_tag(
            "capability:exogenous"
        ) or self.residual_forecaster.get_tag("capability:exogenous")

        miss = self.base_forecaster.get_tag(
            "capability:missing_values"
        ) and self.residual_forecaster.get_tag("capability:missing_values")

        pred_int = self.residual_forecaster.get_tag("capability:pred_int")

        in_sample = self.base_forecaster.get_tag(
            "capability:insample"
        ) and self.residual_forecaster.get_tag("capability:insample")

        cat = self.base_forecaster.get_tag(
            "capability:categorical_in_X"
        ) and self.residual_forecaster.get_tag("capability:categorical_in_X")

        pred_int_insample = self.residual_forecaster.get_tag(
            "capability:pred_int:insample"
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

    def _align_like(self, B, A):
        """
        Align forecast output ``B`` to reference ``A`` so they can be added.

        Why this is needed
        ------------------
        Different forecasters in a pipeline may return pandas objects
        (Series/DataFrames) with structurally compatible values but *incompatible*
        index/column metadata:
        - mismatched index/column **names** (e.g. ``None`` vs ``['time']``)
        - different index/column **orders**
        - differing numbers of index levels (flat vs MultiIndex)

        Pandas will refuse arithmetic (``A + B``) when index level names differ,
        even if the labels align, raising errors like:
        ``ValueError: cannot join with no overlapping index names``.

        What this does
        --------------
        * If ``A`` and ``B`` have the same number of index/column levels,
        set the names of ``B`` to match ``A``.
        * Reindex ``B`` to the row/column labels of ``A`` (in order),
        filling with NaN if ``B`` is missing any labels.
        * If levels differ, skip renaming and just try to reindex on labels.

        This ensures arithmetic like ``A + _align_like(B, A)`` works without
        pandas raising alignment errors.

        Notes
        -----
        - Does *not* force MultiIndex levels to match in number, only names.
        - Keeps a copy of ``B`` only if metadata is mutated.
        - Used internally in ``_predict*`` methods when combining outputs
        from base and residual forecasters.
        """
        out = B

        # --- ROW INDEX NAMES ---
        if hasattr(B, "index") and hasattr(A, "index"):
            nB = getattr(B.index, "nlevels", 1)
            nA = getattr(A.index, "nlevels", 1)

            if nB == nA and B.index.names != A.index.names:
                out = out.copy()
                out.index = out.index.set_names(A.index.names)

            # reindex rows to A only if labels overlap
            if not B.index.equals(A.index):
                out = out.reindex(A.index)

        # --- COLUMN INDEX NAMES ---
        if hasattr(B, "columns") and hasattr(A, "columns"):
            nB = getattr(B.columns, "nlevels", 1)
            nA = getattr(A.columns, "nlevels", 1)

            if nB == nA and B.columns.names != A.columns.names:
                if out is B:
                    out = out.copy()
                out.columns = out.columns.set_names(A.columns.names)

            # reindex cols to A only if labels overlap
            if not B.columns.equals(A.columns):
                out = out.reindex(columns=A.columns)

        return out

    def _predict(self, fh=None, X=None):
        """
        Forecast = base forecast + residual forecast.

        1. Use clone B of base_forecaster to obtain a prediction y_pred_base
        2. Use residual_forecaster clone to obtain a prediction y_pred_resid
        3. Return y_pred_base + y_pred_resid
        """
        y_base = self.base_future_.predict(fh=fh, X=X)
        y_hat = y_base
        for _, f in getattr(self, "_resid_futures_", []):
            y_add = f.predict(fh=fh, X=X)
            y_add = self._align_like(y_add, y_base).fillna(0)
            y_hat = y_hat + y_add
        return y_hat

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Combine prediction intervals from base and residual models."""
        y_shift = self.base_future_.predict(fh=fh, X=X)
        if getattr(self, "_resid_futures_", None):
            for _, f in self._resid_futures_[:-1]:
                y_add = f.predict(fh=fh, X=X)
                y_shift = y_shift + self._align_like(y_add, y_shift).fillna(0)

            _, f_last = self._resid_futures_[-1]
            I_last = f_last.predict_interval(fh=fh, X=X, coverage=coverage)

            # align the shift (Series/DataFrame) to the interval DataFrame
            y_shift_aligned = self._align_like(y_shift, I_last).fillna(0)
            return I_last + y_shift_aligned

        return self.base_future_.predict_interval(fh=fh, X=X, coverage=coverage)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Combine arbitrary quantile forecasts."""
        y_shift = self.base_future_.predict(fh=fh, X=X)
        if getattr(self, "_resid_futures_", None):
            for _, f in self._resid_futures_[:-1]:
                y_add = f.predict(fh=fh, X=X)
                y_shift = y_shift + self._align_like(y_add, y_shift).fillna(0)

            _, f_last = self._resid_futures_[-1]
            Q_last = f_last.predict_quantiles(fh=fh, X=X, alpha=alpha)

            y_shift_aligned = self._align_like(y_shift, Q_last).fillna(0)
            return Q_last + y_shift_aligned

        return self.base_future_.predict_quantiles(fh=fh, X=X, alpha=alpha)

    def _predict_var(self, fh, X=None, cov=False):
        """Combine predictive variances (or full covariances)."""
        if getattr(self, "_resid_futures_", None):
            _, f_last = self._resid_futures_[-1]
            V_last = f_last.predict_var(fh=fh, X=X, cov=cov)
            return V_last
        return self.base_future_.predict_var(fh=fh, X=X, cov=cov)

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

        y_base = self.base_future_.predict(fh=fh, X=X)

        if not getattr(self, "_resid_futures_", []):
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        _, f = self._resid_futures_[0]

        # only proceed if residual forecaster supports predict_proba
        if not hasattr(f, "predict_proba"):
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        p_res = f.predict_proba(fh=fh, X=X, marginal=marginal)

        # align base mean with residual distribution index if possible
        if hasattr(y_base, "reindex") and not y_base.index.equals(p_res.index):
            mu = y_base.reindex(p_res.index)
        else:
            mu = y_base

        return MeanScale(d=p_res, mu=mu, sigma=1)

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
            "base_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": NaiveForecaster(strategy="mean"),
        }

        params2 = {
            "base_forecaster": YfromX(
                estimator=LinearRegression(),
                pooling="local",
            ),
            "residual_forecaster": NaiveForecaster(strategy="mean"),
        }

        params3 = {
            "base_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": [
                NaiveForecaster(strategy="last", sp=7),
                NaiveForecaster(strategy="last", sp=12),
            ],
        }

        return [params1, params2, params3]
