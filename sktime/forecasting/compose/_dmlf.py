#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License
"""
Double Machine Learning forecaster for time series.

Implements DML methodology to de-confound exposure effects in forecasting.
"""

__all__ = ["DMLForecaster"]
__author__ = ["geetu040", "XAheli"]

from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import BaseForecaster


class DMLForecaster(BaseForecaster):
    """Double Machine Learning forecaster for causal effect estimation.

    DML forecaster implements the Double Machine Learning methodology for
    causal forecasting with confounder adjustment. It addresses scenarios where
    exogenous variables (exposures) are confounded, leading to biased estimates
    of their true causal impact on the target variable.

    The forecaster follows a three-step process:
    1. Model the outcome from confounders to get outcome residuals
    2. Model the exposure from confounders to get exposure residuals
    3. Model the final causal relationship between the residuals

    This approach enables unbiased estimation of causal effects by removing
    confounding influences through residualization before estimating the
    treatment effect.

    Parameters
    ----------
    forecaster_y : sktime forecaster
        Forecaster to model the outcome (y) from confounder variables (X_conf).
        Used to control for confounding effects on the target variable.

    forecaster_ex : sktime forecaster
        Forecaster to model exposure variables (X_ex) from confounder
        variables (X_conf). Used to control for confounding effects on
        the treatment variables.

    forecaster_res : sktime forecaster, default=None
        Forecaster to model the final causal relationship between residuals.
        If None, uses LinearRegression wrapped in make_reduction.
        Should typically be a simple linear model for interpretability.

    exposure_vars : list of str
        Column names in X that should be treated as exposure/treatment variables.
        These are the variables whose causal effects we want to estimate.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.dml import DMLForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> y, X = load_longley()
    >>> # Assume 'GNP' is our exposure variable of interest
    >>> exposure_vars = ['GNP']
    >>>
    >>> # Set up forecasters for nuisance functions
    >>> forecaster_y = make_reduction(RandomForestRegressor(n_estimators=10))
    >>> forecaster_ex = make_reduction(RandomForestRegressor(n_estimators=10))
    >>>
    >>> # Create DML forecaster
    >>> dml_forecaster = DMLForecaster(
    ...     forecaster_y=forecaster_y,
    ...     forecaster_ex=forecaster_ex,
    ...     exposure_vars=exposure_vars
    ... )
    >>>
    >>> # Fit and predict
    >>> fh = [1, 2, 3]
    >>> dml_forecaster.fit(y, X=X, fh=fh)
    >>> y_pred = dml_forecaster.predict(fh, X=X)

    Notes
    -----
    The current implementation uses in-sample residuals during fitting,
    similar to ResidualBoostingForecaster. Future versions may incorporate
    cross-fitting for enhanced statistical robustness.

    The methodology is based on the Double Machine Learning framework,
    which provides theoretical guarantees for unbiased causal estimation
    even when nuisance functions are estimated with flexible ML models.

    References
    ----------
    .. [1] Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
           Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased
           machine learning for treatment and structural parameters.
    """

    _tags = {
        "authors": ["geetu040", "XAheli"],
        "maintainers": ["geetu040", "XAheli"],
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": True,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
    }

    def __init__(
        self, forecaster_y, forecaster_ex, forecaster_res=None, exposure_vars=None
    ):
        self.forecaster_y = forecaster_y
        self.forecaster_ex = forecaster_ex
        self.forecaster_res = forecaster_res
        self.exposure_vars = exposure_vars

        # fitted copies of user passed forecasters
        self.forecaster_y_ = None
        self.forecaster_ex_ = None
        self.forecaster_res_ = None

        super().__init__()

        # Handle null forecaster_res
        if self.forecaster_res is None:
            from sktime.forecasting.compose import make_reduction

            self.forecaster_res = make_reduction(
                LinearRegression(), strategy="recursive"
            )

        # Handle null exposure_vars
        if self.exposure_vars is None:
            self.exposure_vars = []

        # Update tags based on component forecasters
        self._update_tags_from_components()

    def _update_tags_from_components(self):
        """Update forecaster tags based on component capabilities."""
        exog = (
            self.forecaster_y.get_tag("capability:exogenous")
            or self.forecaster_ex.get_tag("capability:exogenous")
            or self.forecaster_res.get_tag("capability:exogenous")
        )

        assert self.forecaster_y.get_tag("capability:insample"), (
            "these have to do insample in all case"
        )
        assert self.forecaster_ex.get_tag("capability:insample"), (
            "these have to do insample in all case"
        )
        in_sample = self.forecaster_res.get_tag("capability:insample")

        pred_int = (
            self.forecaster_y.get_tag("capability:pred_int")
            # and self.forecaster_ex.get_tag("capability:pred_int")
            and self.forecaster_res.get_tag("capability:pred_int")
        )

        pred_int_insample = (
            self.forecaster_y.get_tag("capability:pred_int:insample")
            # and self.forecaster_ex.get_tag("capability:pred_int:insample")
            and self.forecaster_res.get_tag("capability:pred_int:insample")
        )

        miss = (
            self.forecaster_y.get_tag("capability:missing_values")
            and self.forecaster_ex.get_tag("capability:missing_values")
            and self.forecaster_res.get_tag("capability:missing_values")
        )

        cat = (
            self.forecaster_y.get_tag("capability:categorical_in_X")
            and self.forecaster_ex.get_tag("capability:categorical_in_X")
            and self.forecaster_res.get_tag("capability:categorical_in_X")
        )

        self.set_tags(
            **{
                "capability:exogenous": exog,
                "capability:insample": in_sample,
                "capability:pred_int": pred_int,
                "capability:pred_int:insample": pred_int_insample,
                "capability:missing_values": miss,
                "capability:categorical_in_X": cat,
            }
        )

    def _split_exogenous_data(self, X=None):
        """Split exogenous data into exposure and confounder variables.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous variables

        Returns
        -------
        X_ex : pd.DataFrame
            Exposure variables
        X_conf : pd.DataFrame
            Confounder variables
        """
        if X is None:
            return None, None

        # Check that all exposure variables exist in X
        missing_vars = set(self.exposure_vars) - set(X.columns)
        if missing_vars:
            raise ValueError(
                f"Exposure variables {missing_vars} not found in X columns: "
                f"{list(X.columns)}"
            )

        X_ex = X[self.exposure_vars].copy()
        X_conf = X.drop(columns=self.exposure_vars).copy()

        if X_ex.empty:
            X_ex = None

        if X_conf.empty:
            X_conf = None

        return X_ex, X_conf

    def _fit(self, y, X=None, fh=None):
        """Fit the DML forecaster.

        Implements the DML fitting procedure:
        1. Split X into exposure (X_ex) and confounder (X_conf) variables
        2. Fit forecaster_y on y and X_conf to get outcome residuals
        3. Fit forecaster_ex on X_ex and X_conf to get exposure residuals
        4. Fit forecaster_res on the residual relationship
        5. Fit final versions for prediction
        """
        X_ex, X_conf = self._split_exogenous_data(X)

        forecaster_y_insample = clone(self.forecaster_y)
        forecaster_y_insample.fit(y=y, X=X_conf, fh=y.index)
        y_res = forecaster_y_insample.predict_residuals()

        if X_ex is None:
            X_ex_res = None
        else:
            forecaster_ex_insample = clone(self.forecaster_ex)
            forecaster_ex_insample.fit(y=X_ex, X=X_conf, fh=y.index)
            X_ex_res = forecaster_ex_insample.predict_residuals()

        self.forecaster_res_ = clone(self.forecaster_res)
        self.forecaster_res_.fit(y=y_res, X=X_ex_res, fh=fh)

        self.forecaster_y_ = clone(self.forecaster_y)
        self.forecaster_y_.fit(y=y, X=X_conf, fh=fh)

        if X_ex is not None:
            self.forecaster_ex_ = clone(self.forecaster_ex)
            self.forecaster_ex_.fit(y=X_ex, X=X_conf, fh=fh)

    def _predict(self, fh=None, X=None):
        """Generate DML forecasts.

        Prediction logic:
        1. Split X_new into exposure and confounder variables
        2. Get base forecast from confounders: y_pred_conf
        3. Get de-confounded exposure: X_ex_deconf
        4. Get causal forecast component: y_pred_causal
        5. Return combined forecast: y_pred_conf + y_pred_causal
        """
        X_ex, X_conf = self._split_exogenous_data(X)

        if X_ex is None:
            X_ex_res = None
        else:
            X_ex_res = self.forecaster_ex_.predict_residuals(y=X_ex, X=X_conf)

        pred_base = self.forecaster_y_.predict(fh=fh, X=X_conf)
        pred_res = self.forecaster_res_.predict(fh=fh, X=X_ex_res)

        return pred_base + pred_res

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Generate prediction intervals for DML forecasts."""
        X_ex, X_conf = self._split_exogenous_data(X)

        if X_ex is None:
            X_ex_res = None
        else:
            X_ex_res = self.forecaster_ex_.predict_residuals(y=X_ex, X=X_conf)

        pred_int_base = self.forecaster_y_.predict_interval(
            fh=fh, X=X_conf, coverage=coverage
        )
        pred_int_res = self.forecaster_res_.predict_interval(
            fh=fh, X=X_ex_res, coverage=coverage
        )

        return pred_int_base + pred_int_res

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Generate quantile forecasts for DML."""
        X_ex, X_conf = self._split_exogenous_data(X)

        if X_ex is None:
            X_ex_res = None
        else:
            X_ex_res = self.forecaster_ex_.predict_residuals(y=X_ex, X=X_conf)

        pred_quantiles_base = self.forecaster_y_.predict_quantiles(
            fh=fh, X=X_conf, alpha=alpha
        )
        pred_quantiles_res = self.forecaster_res_.predict_quantiles(
            fh=fh, X=X_ex_res, alpha=alpha
        )

        return pred_quantiles_base + pred_quantiles_res

    def _predict_var(self, fh, X=None, cov=False):
        """Generate predictive variances for DML forecasts."""
        X_ex, X_conf = self._split_exogenous_data(X)

        if X_ex is None:
            X_ex_res = None
        else:
            X_ex_res = self.forecaster_ex_.predict_residuals(y=X_ex, X=X_conf)

        pred_var_base = self.forecaster_y_.predict_var(fh=fh, X=X_conf, cov=cov)
        pred_var_res = self.forecaster_res_.predict_var(fh=fh, X=X_ex_res, cov=cov)

        return pred_var_base + pred_var_res

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

        X_ex, X_conf = self._split_exogenous_data(X)

        if X_ex is None:
            X_ex_res = None
        else:
            X_ex_res = self.forecaster_ex_.predict_residuals(y=X_ex, X=X_conf)

        pred_base = self.forecaster_y_.predict(fh=fh, X=X_conf)
        pred_proba_res = self.forecaster_res_.predict_proba(
            fh=fh, X=X_ex_res, marginal=marginal
        )

        return MeanScale(
            d=pred_proba_res,
            mu=pred_base,
            sigma=1,
            index=pred_proba_res.index,
            columns=pred_proba_res.columns,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create test instances of the estimator.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.naive import NaiveForecaster

        # Basic test parameters
        params1 = {
            "forecaster_y": NaiveForecaster(strategy="mean"),
            "forecaster_ex": NaiveForecaster(strategy="mean"),
            "forecaster_res": make_reduction(LinearRegression()),
            "exposure_vars": ["var_0"],  # Standard sktime test data variable
        }

        # More complex test parameters
        params2 = {
            "forecaster_y": make_reduction(
                RandomForestRegressor(n_estimators=5, random_state=42)
            ),
            "forecaster_ex": make_reduction(LinearRegression()),
            "forecaster_res": NaiveForecaster(strategy="last"),
            "exposure_vars": ["var_0"],
        }

        return [params1, params2]

    def get_fitted_params(self, deep=True):
        """Get fitted parameters from the forecaster.

        Returns
        -------
        fitted_params : dict
            Dictionary of fitted parameters, includes causal effect estimates
            and fitted component forecasters.
        """
        fitted_params = super().get_fitted_params(deep=deep)

        if hasattr(self, "forecaster_res_"):
            # Try to extract causal effect estimate if available
            try:
                if hasattr(self.forecaster_res_, "estimator_"):
                    estimator = self.forecaster_res_.estimator_
                    if hasattr(estimator, "coef_"):
                        fitted_params["causal_effect"] = estimator.coef_
                    if hasattr(estimator, "intercept_"):
                        fitted_params["intercept"] = estimator.intercept_
            except AttributeError:
                pass

            fitted_params.update(
                {
                    "y_residuals_": self.y_residuals_,
                    "X_ex_residuals_": self.X_ex_residuals_,
                    "forecaster_y_final_": self.forecaster_y_final_,
                    "forecaster_ex_final_": self.forecaster_ex_final_,
                    "forecaster_res_": self.forecaster_res_,
                }
            )

        return fitted_params
