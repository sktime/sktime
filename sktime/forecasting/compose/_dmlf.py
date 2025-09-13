#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License
"""
Double Machine Learning forecaster for time series.

Implements DML methodology to de-confound exposure effects in forecasting.
"""

__all__ = ["DMLForecaster"]
__author__ = ["XAheli"]

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


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
        "authors": ["XAheli"],
        "maintainers": ["XAheli"],
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "ignores-exogeneous-X": False,  # Requires X for causal inference
        "capability:pred_int": True,
        "capability:pred_var": True,
        "capability:pred_quantiles": True,
        "capability:insample": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "fit_is_empty": False,
    }

    def __init__(self, forecaster_y, forecaster_ex, exposure_vars, forecaster_res=None):
        self.forecaster_y = forecaster_y
        self.forecaster_ex = forecaster_ex
        self.forecaster_res = forecaster_res
        self.exposure_vars = exposure_vars

        super().__init__()

        # Set default forecaster_res if not provided
        if self.forecaster_res is None:
            from sktime.forecasting.compose import make_reduction

            self.forecaster_res = make_reduction(
                LinearRegression(), strategy="recursive"
            )

        # Validate exposure_vars
        if self.exposure_vars is None or len(self.exposure_vars) == 0:
            raise ValueError(
                "exposure_vars must be specified and contain at least one variable"
            )

        # Update tags based on component forecasters
        self._update_tags_from_components()

    def _update_tags_from_components(self):
        """Update forecaster tags based on component capabilities."""
        # Missing values capability - all components must support it
        miss_y = self.forecaster_y.get_tag("capability:missing_values", False)
        miss_ex = self.forecaster_ex.get_tag("capability:missing_values", False)
        miss_res = self.forecaster_res.get_tag("capability:missing_values", False)

        # Prediction interval capability - all components must support it
        pred_int_y = self.forecaster_y.get_tag("capability:pred_int", False)
        pred_int_ex = self.forecaster_ex.get_tag("capability:pred_int", False)
        pred_int_res = self.forecaster_res.get_tag("capability:pred_int", False)

        # In-sample prediction capability
        insample_y = self.forecaster_y.get_tag("capability:insample", False)
        insample_ex = self.forecaster_ex.get_tag("capability:insample", False)
        insample_res = self.forecaster_res.get_tag("capability:insample", False)

        self.set_tags(
            **{
                "capability:missing_values": miss_y and miss_ex and miss_res,
                "capability:pred_int": pred_int_y and pred_int_ex and pred_int_res,
                "capability:insample": insample_y or insample_ex or insample_res,
            }
        )

    def _split_exogenous_data(self, X):
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
            raise ValueError("X cannot be None for DML forecasting")

        # Check that all exposure variables exist in X
        missing_vars = set(self.exposure_vars) - set(X.columns)
        if missing_vars:
            raise ValueError(
                f"Exposure variables {missing_vars} not found in X columns: "
                f"{list(X.columns)}"
            )

        X_ex = X[self.exposure_vars].copy()
        X_conf = X.drop(columns=self.exposure_vars).copy()

        # Handle case where no confounders remain
        if X_conf.shape[1] == 0:
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
        # Split exogenous data
        X_ex, X_conf = self._split_exogenous_data(X)

        # Step 1: Fit outcome model for residuals (clone A)
        self.forecaster_y_insample_ = clone(self.forecaster_y)
        self.forecaster_y_insample_.fit(y, X=X_conf, fh=fh)

        # Get in-sample outcome residuals
        if isinstance(y.index, pd.MultiIndex):
            time_idx = y.index.get_level_values(-1).unique()
        else:
            time_idx = y.index

        insample_fh = ForecastingHorizon(time_idx, is_relative=False)
        y_pred_insample = self.forecaster_y_insample_.predict(fh=insample_fh, X=X_conf)
        self.y_residuals_ = y - y_pred_insample

        # Step 2: Fit exposure model for residuals
        self.forecaster_ex_insample_ = clone(self.forecaster_ex)
        self.forecaster_ex_insample_.fit(X_ex, X=X_conf, fh=fh)

        # Get in-sample exposure residuals
        X_ex_pred_insample = self.forecaster_ex_insample_.predict(
            fh=insample_fh, X=X_conf
        )
        self.X_ex_residuals_ = X_ex - X_ex_pred_insample

        # Step 3: Fit residual relationship
        self.forecaster_res_ = clone(self.forecaster_res)
        self.forecaster_res_.fit(self.y_residuals_, X=self.X_ex_residuals_, fh=fh)

        # Step 4: Fit final versions for prediction
        self.forecaster_y_final_ = clone(self.forecaster_y)
        self.forecaster_y_final_.fit(y, X=X_conf, fh=fh)

        self.forecaster_ex_final_ = clone(self.forecaster_ex)
        self.forecaster_ex_final_.fit(X_ex, X=X_conf, fh=fh)

        return self

    def _predict(self, fh=None, X=None):
        """Generate DML forecasts.

        Prediction logic:
        1. Split X_new into exposure and confounder variables
        2. Get base forecast from confounders: y_pred_conf
        3. Get de-confounded exposure: X_ex_deconf
        4. Get causal forecast component: y_pred_causal
        5. Return combined forecast: y_pred_conf + y_pred_causal
        """
        # Split exogenous data
        X_ex_new, X_conf_new = self._split_exogenous_data(X)

        # Base forecast from confounders
        y_pred_conf = self.forecaster_y_final_.predict(fh=fh, X=X_conf_new)

        # De-confounded exposure
        X_ex_pred = self.forecaster_ex_final_.predict(fh=fh, X=X_conf_new)
        X_ex_deconf = X_ex_new - X_ex_pred

        # Causal forecast component
        y_pred_causal = self.forecaster_res_.predict(fh=fh, X=X_ex_deconf)

        # Combined forecast
        return y_pred_conf + y_pred_causal

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Generate prediction intervals for DML forecasts."""
        # Split exogenous data
        X_ex_new, X_conf_new = self._split_exogenous_data(X)

        # Base prediction intervals
        pred_int_conf = self.forecaster_y_final_.predict_interval(
            fh=fh, X=X_conf_new, coverage=coverage
        )

        # De-confounded exposure
        X_ex_pred = self.forecaster_ex_final_.predict(fh=fh, X=X_conf_new)
        X_ex_deconf = X_ex_new - X_ex_pred

        # Causal prediction intervals
        pred_int_causal = self.forecaster_res_.predict_interval(
            fh=fh, X=X_ex_deconf, coverage=coverage
        )

        # Combine intervals (assumes independence)
        return pred_int_conf + pred_int_causal

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Generate quantile forecasts for DML."""
        # Split exogenous data
        X_ex_new, X_conf_new = self._split_exogenous_data(X)

        # Base quantiles
        q_conf = self.forecaster_y_final_.predict_quantiles(
            fh=fh, X=X_conf_new, alpha=alpha
        )

        # De-confounded exposure
        X_ex_pred = self.forecaster_ex_final_.predict(fh=fh, X=X_conf_new)
        X_ex_deconf = X_ex_new - X_ex_pred

        # Causal quantiles
        q_causal = self.forecaster_res_.predict_quantiles(
            fh=fh, X=X_ex_deconf, alpha=alpha
        )

        # Combine quantiles
        return q_conf + q_causal

    def _predict_var(self, fh, X=None, cov=False):
        """Generate predictive variances for DML forecasts."""
        # Split exogenous data
        X_ex_new, X_conf_new = self._split_exogenous_data(X)

        # Base variance
        var_conf = self.forecaster_y_final_.predict_var(fh=fh, X=X_conf_new, cov=cov)

        # De-confounded exposure
        X_ex_pred = self.forecaster_ex_final_.predict(fh=fh, X=X_conf_new)
        X_ex_deconf = X_ex_new - X_ex_pred

        # Causal variance
        var_causal = self.forecaster_res_.predict_var(fh=fh, X=X_ex_deconf, cov=cov)

        # Combine variances (assumes independence)
        return var_conf + var_causal

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
