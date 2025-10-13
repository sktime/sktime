# copyright: sktime developers, BSD-3-Clause License
"""
Double Machine Learning forecaster for time series.

Implements Double Machine Learning methodology to de-confound exposure
effects in forecasting.
"""

__all__ = ["DoubleMLForecaster"]
__author__ = ["geetu040", "XAheli"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class DoubleMLForecaster(BaseForecaster):
    """Double Machine Learning Forecaster for causal effect estimation.

    DoubleMLForecaster implements the Double Machine Learning methodology for
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
    outcome_forecaster : sktime forecaster
        Forecaster to model the outcome (y) from confounder variables (X_confounder).
        Used to control for confounding effects on the target variable.

    treatment_forecaster : sktime forecaster
        Forecaster to model exposure variables (X_exposure) from confounder
        variables (X_confounder). Used to control for confounding effects on
        the treatment variables.

    residual_forecaster : sktime forecaster, default=None
        Forecaster to model the final causal relationship between residuals.
        If None, uses LinearRegression wrapped in make_reduction.
        Should typically be a simple linear model for interpretability.

    exposure_vars : list of str
        Column names in X that should be treated as exposure/treatment variables.
        These are the variables whose causal effects we want to estimate.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.compose import DoubleMLForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)
    >>>
    >>> # Assume 'GNP' is our exposure variable of interest
    >>> exposure_vars = ['GNP']
    >>>
    >>> # Set up forecasters for nuisance functions
    >>> outcome_forecaster = NaiveForecaster()
    >>> treatment_forecaster = NaiveForecaster()
    >>>
    >>> # Create DoubleMLForecaster
    >>> dml_forecaster = DoubleMLForecaster(
    ...     outcome_forecaster=outcome_forecaster,
    ...     treatment_forecaster=treatment_forecaster,
    ...     exposure_vars=exposure_vars
    ... )
    >>>
    >>> # Fit and predict
    >>> fh = [1, 2, 3]
    >>> dml_forecaster.fit(y_train, X=X_train, fh=fh)
    DoubleMLForecaster(exposure_vars=['GNP'], outcome_forecaster=NaiveForecaster(),
                       residual_forecaster=RecursiveTabularRegressionForecaster(estimator=LinearRegression()),
                       treatment_forecaster=NaiveForecaster())
    >>> y_pred = dml_forecaster.predict(X=X_test)

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
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        outcome_forecaster,
        treatment_forecaster,
        residual_forecaster=None,
        exposure_vars=None,
    ):
        self.outcome_forecaster = outcome_forecaster
        self.treatment_forecaster = treatment_forecaster
        self.residual_forecaster = residual_forecaster
        self.exposure_vars = exposure_vars

        # fitted copies of user passed forecasters
        self.outcome_forecaster_ = None
        self.treatment_forecaster_ = None
        self.residual_forecaster_ = None

        super().__init__()

        # Handle null residual_forecaster
        if self.residual_forecaster is None:
            from sktime.forecasting.compose import make_reduction

            self.residual_forecaster = make_reduction(
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
            self.outcome_forecaster.get_tag("capability:exogenous")
            or self.treatment_forecaster.get_tag("capability:exogenous")
            or self.residual_forecaster.get_tag("capability:exogenous")
        )

        # self.treatment_forecaster and self.outcome_forecaster must support insample
        # under normal circumstances as well
        in_sample = self.residual_forecaster.get_tag("capability:insample")

        pred_int = (
            self.outcome_forecaster.get_tag("capability:pred_int")
            # and self.treatment_forecaster.get_tag("capability:pred_int")
            and self.residual_forecaster.get_tag("capability:pred_int")
        )

        pred_int_insample = (
            self.outcome_forecaster.get_tag("capability:pred_int:insample")
            # and self.treatment_forecaster.get_tag("capability:pred_int:insample")
            and self.residual_forecaster.get_tag("capability:pred_int:insample")
        )

        miss = (
            self.outcome_forecaster.get_tag("capability:missing_values")
            and self.treatment_forecaster.get_tag("capability:missing_values")
            and self.residual_forecaster.get_tag("capability:missing_values")
        )

        cat = (
            self.outcome_forecaster.get_tag("capability:categorical_in_X")
            and self.treatment_forecaster.get_tag("capability:categorical_in_X")
            and self.residual_forecaster.get_tag("capability:categorical_in_X")
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
        X_exposure : pd.DataFrame
            Exposure variables
        X_confounder : pd.DataFrame
            Confounder variables
        """
        if X is None:
            return None, None

        # Check that all exposure variables exist in X
        missing_vars = set(self.exposure_vars) - set(X.columns)
        found_vars = [i for i in self.exposure_vars if i in X.columns]
        if missing_vars:
            from sktime.utils.warnings import warn

            warn(
                f"Exposure variables: {list(missing_vars)} "
                f"not found in X columns: {list(X.columns)}. "
                f"Proceeding with available exposure variables: {found_vars}.",
                category=UserWarning,
                obj=self,
            )

        X_exposure = X[found_vars].copy()
        X_confounder = X.drop(columns=found_vars).copy()

        if X_exposure.empty:
            X_exposure = None

        if X_confounder.empty:
            X_confounder = None

        return X_exposure, X_confounder

    def _fit(self, y, X=None, fh=None):
        """Fit the DoubleMLForecaster.

        Implements the DoubleMLForecaster fitting procedure:
        1. Split X into exposure (X_exposure) and confounder (X_confounder) variables
        2. Fit outcome_forecaster on y and X_confounder to get outcome residuals
        3. Fit treatment_forecaster on X_exposure and X_confounder
           to get exposure residuals
        4. Fit residual_forecaster on the residual relationship
        5. Fit final versions for prediction
        """
        X_exposure, X_confounder = self._split_exogenous_data(X)

        # Forecast insample
        if isinstance(y.index, pd.MultiIndex):
            time_idx = y.index.get_level_values(-1).unique()
        else:
            time_idx = y.index
        insample_fh = ForecastingHorizon(time_idx, is_relative=False)

        outcome_forecaster_insample = clone(self.outcome_forecaster)
        outcome_forecaster_insample.fit(y=y, X=X_confounder, fh=insample_fh)
        y_pred = outcome_forecaster_insample.predict(X=X_confounder)
        y_res = y - y_pred

        if X_exposure is None:
            X_exposure_res = None
        else:
            treatment_forecaster_insample = clone(self.treatment_forecaster)
            treatment_forecaster_insample.fit(
                y=X_exposure, X=X_confounder, fh=insample_fh
            )
            X_exposure_pred = treatment_forecaster_insample.predict(X=X_confounder)
            X_exposure_res = X_exposure - X_exposure_pred

        self.residual_forecaster_ = clone(self.residual_forecaster)
        self.residual_forecaster_.fit(y=y_res, X=X_exposure_res, fh=fh)

        self.outcome_forecaster_ = clone(self.outcome_forecaster)
        self.outcome_forecaster_.fit(y=y, X=X_confounder, fh=fh)

        if X_exposure is not None:
            self.treatment_forecaster_ = clone(self.treatment_forecaster)
            self.treatment_forecaster_.fit(y=X_exposure, X=X_confounder, fh=fh)

    def _compute_X_exposure_res(self, X_exposure=None, X_confounder=None, fh=None):
        if X_exposure is None:
            return None

        X_exposure_pred = self.treatment_forecaster_.predict(fh=fh, X=X_confounder)
        X_exposure_aligned = X_exposure.loc[X_exposure_pred.index]
        X_exposure_res = X_exposure_aligned - X_exposure_pred

        return X_exposure_res

    def _predict(self, fh=None, X=None):
        """Generate DoubleMLForecasts.

        Prediction logic:
        1. Split X_new into exposure and confounder variables
        2. Get base forecast from confounders: y_pred_conf
        3. Get de-confounded exposure: X_exposure_deconf
        4. Get causal forecast component: y_pred_causal
        5. Return combined forecast: y_pred_conf + y_pred_causal
        """
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_base = self.outcome_forecaster_.predict(fh=fh, X=X_confounder)
        pred_res = self.residual_forecaster_.predict(fh=fh, X=X_exposure_res)

        return pred_base + pred_res

    def _add_det_to_proba(self, y_proba, y_pred):
        """Add multiindex columns to probabilistic forecasts."""
        y_proba = y_proba.copy()
        for col in y_proba.columns:
            var = col[0]
            y_proba[col] = y_proba[col] + y_pred[var]
        return y_proba

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Generate prediction intervals for DoubleMLForecaster."""
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_base = self.outcome_forecaster_.predict(fh=fh, X=X_confounder)
        pred_int_res = self.residual_forecaster_.predict_interval(
            fh=fh, X=X_exposure_res, coverage=coverage
        )

        return self._add_det_to_proba(pred_int_res, pred_base)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Generate quantile forecasts for DoubleMLForecaster."""
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_base = self.outcome_forecaster_.predict(fh=fh, X=X_confounder)
        pred_quantiles_res = self.residual_forecaster_.predict_quantiles(
            fh=fh, X=X_exposure_res, alpha=alpha
        )

        return self._add_det_to_proba(pred_quantiles_res, pred_base)

    def _predict_var(self, fh, X=None, cov=False):
        """Generate predictive variances for DoubleMLForecaster."""
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_var_res = self.residual_forecaster_.predict_var(
            fh=fh, X=X_exposure_res, cov=cov
        )

        return pred_var_res

    def _predict_proba(self, fh, X=None, marginal=True):
        """Combine full distribution forecasts from component models."""
        if not _check_soft_dependencies("skpro", severity="none"):
            from sktime.utils.warnings import warn

            warn(
                "DoubleMLForecaster.predict_proba: optional "
                "dependency 'skpro' not found. "
                "Falling back to the default normal approximation via BaseForecaster. "
                "Install 'skpro' to enable exact shifted-distribution composition.",
                category=UserWarning,
                obj=self,
            )
            return super()._predict_proba(fh=fh, X=X, marginal=marginal)

        from skpro.distributions import MeanScale

        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_base = self.outcome_forecaster_.predict(fh=fh, X=X_confounder)
        pred_proba_res = self.residual_forecaster_.predict_proba(
            fh=fh, X=X_exposure_res, marginal=marginal
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
            "outcome_forecaster": NaiveForecaster(strategy="last"),
            "treatment_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": make_reduction(LinearRegression(), window_length=3),
            "exposure_vars": [0, "foo"],
        }

        # More complex test parameters
        params2 = {
            "outcome_forecaster": NaiveForecaster(strategy="last"),
            "treatment_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": make_reduction(
                RandomForestRegressor(n_estimators=5, random_state=42),
                window_length=3,
            ),
            "exposure_vars": [0, "foo"],
        }

        return [params1, params2]
