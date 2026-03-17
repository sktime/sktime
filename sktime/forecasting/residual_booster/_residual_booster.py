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

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class ResidualBoostingForecaster(BaseForecaster):
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
    residual_forecaster : sktime forecaster
        Model trained on the base model's in-sample residuals.

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
        Fit base forecaster and residual forecaster.

        1. Fit clone A of base_forecaster to X, y, and compute in-sample
           forecast residuals r
        2. Fit clone B of base_forecaster to X, y, with fh
        3. Fit clone of residual_forecaster to X, r
        """
        # clone A: fit on (y,X) to obtain in-sample residuals
        # 1. in-sample residuals
        self.base_insample_ = clone(self.base_forecaster).fit(y, X, fh)

        # Forecast insample
        if isinstance(y.index, pd.MultiIndex):
            time_idx = y.index.get_level_values(-1).unique()
        else:
            time_idx = y.index
        insample_fh = ForecastingHorizon(time_idx, is_relative=False)

        insample_preds = self.base_insample_.predict(fh=insample_fh, X=X)

        residuals = y - insample_preds

        # clone B: fit a fresh copy that knows the final fh
        # 2. future base model with final fh
        self.base_future_ = clone(self.base_forecaster).fit(y, X, fh)

        # clone C: fit residual model on errors
        # 3. residual model
        self.residual_forecaster_ = clone(self.residual_forecaster).fit(
            residuals, X, fh
        )
        return self

    def _predict(self, fh=None, X=None):
        """
        Forecast = base forecast + residual forecast.

        1. Use clone B of base_forecaster to obtain a prediction y_pred_base
        2. Use residual_forecaster clone to obtain a prediction y_pred_resid
        3. Return y_pred_base + y_pred_resid
        """
        y_base = self.base_future_.predict(fh=fh, X=X)
        y_resid = self.residual_forecaster_.predict(fh=fh, X=X)
        return y_base + y_resid

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Combine prediction intervals from base and residual models."""
        i_base = self.base_future_.predict(fh=fh, X=X)
        i_res = self.residual_forecaster_.predict_interval(
            fh=fh, X=X, coverage=coverage
        )
        return self._add_det_to_proba(i_res, i_base)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Combine arbitrary quantile forecasts."""
        q_base = self.base_future_.predict(fh=fh, X=X)
        q_res = self.residual_forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)
        return self._add_det_to_proba(q_res, q_base)

    def _predict_var(self, fh, X=None, cov=False):
        """Combine predictive variances (or full covariances)."""
        v_res = self.residual_forecaster_.predict_var(fh=fh, X=X, cov=cov)
        return v_res

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
        p_res = self.residual_forecaster_.predict_proba(fh=fh, X=X, marginal=marginal)

        return MeanScale(
            d=p_res, mu=y_base, sigma=1, index=p_res.index, columns=p_res.columns
        )

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

        return [params1, params2]
