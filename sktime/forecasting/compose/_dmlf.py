# copyright: sktime developers, BSD-3-Clause License
"""Implementation of Double Machine Learning methodology for forecasting."""

__all__ = ["DoubleMLForecaster"]
__author__ = ["geetu040", "XAheli"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class DoubleMLForecaster(BaseForecaster):
    """Double Machine Learning forecaster for causal time-series forecasting.

    Implements the Double Machine Learning (DML) framework [1]_ for time-series,
    enabling deconfounded estimation of causal effects from observational data.

    The forecaster uses a three-step residualization process to separate
    causal effects from confounding influences:

    **Fit procedure**

    1. Split exogenous data ``X`` into exposure variables ``X_exposure`` and
       confounder variables ``X_confounder``.

    2. Fit the outcome forecaster on ``(y | X_confounder)`` to obtain residuals
       ``y_res = y - y_pred``.

    3. Fit the treatment forecaster on ``(X_exposure | X_confounder)`` to
       obtain residuals ``X_exposure_res = X_exposure - X_exposure_pred``.

    4. Fit the residual forecaster on ``(y_res | X_exposure_res)`` to learn the
       deconfounded causal relationship.

    5. Refit the outcome and treatment forecasters on the full training data
       for use during prediction.

    **Predict procedure**

    1. Split new exogenous data ``X`` into ``X_exposure`` and ``X_confounder``.

    2. Compute the base (confounder-driven) forecast:
       ``y_pred_base = outcome_forecaster.predict(X_confounder)``.

    3. Compute the residualized exposures:
       ``X_exposure_pred = treatment_forecaster.predict(X_confounder)``.
       ``X_exposure_res = X_exposure - X_exposure_pred``.

    4. Compute the causal (residual) forecast:
       ``y_pred_res = residual_forecaster.predict(X=X_exposure_res)``.

    5. Combine both components to obtain the final prediction:
       ``y_pred = y_pred_base + y_pred_res``.

    Parameters
    ----------
    outcome_forecaster : sktime forecaster
        Base forecaster modeling the outcome variable conditional on
        confounders.

    treatment_forecaster : sktime forecaster
        Forecaster modeling the exposure variables conditional on confounders.

    residual_forecaster : sktime forecaster, optional (default=None)
        Forecaster modeling the residual (deconfounded) relationship between
        outcome and treatment. If not provided, a default forecaster is created
        using ``make_reduction(LinearRegression(), strategy="recursive")``,
        a recursive reduction forecaster built from a linear regression model,
        providing a simple and interpretable baseline.

    exposure_vars : list of str, optional (default=None)
        Names of columns in ``X`` representing exposure (treatment) variables.
        The remaining columns are treated as confounders. If None, the model
        assumes there are no explicit exposure variables, using all features as
        confounders and focusing purely on predictive residual correction
        rather than causal effect estimation. In this case, the model behaves
        equivalently to a ``ResidualBoostingForecaster``.

    Attributes
    ----------
    outcome_forecaster_ : sktime forecaster
        Fitted clone of the outcome forecaster.

    treatment_forecaster_ : sktime forecaster
        Fitted clone of the treatment forecaster.

    residual_forecaster_ : sktime forecaster
        Fitted clone of the residual forecaster.

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
    * All provided component forecasters (outcome, treatment, residual) make
      proper use of exogenous data (``X``). In particular, the outcome and
      treatment forecasters must condition on both confounder and exposure
      variables. If a forecaster ignores exogenous inputs, the model reduces to
      a standard residual-based forecaster and loses its causal interpretation.

    * The outcome and treatment forecasters must support in-sample prediction,
      as residuals are computed from fitted values on the training data. If a
      forecaster does not natively support in-sample prediction, it can be
      wrapped using a utility such as ``OosResidualsWrapper`` to enable this
      functionality.

    * The residual forecaster should ideally be a simple interpretable model,
      such as a linear regression or reduced-form model, to preserve
      transparency of causal effect estimates.

    References
    ----------
    .. [1] Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
           Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine
           learning for treatment and structural parameters.
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
        # True if any component supports exogenous data
        exog = (
            self.outcome_forecaster.get_tag("capability:exogenous")
            or self.treatment_forecaster.get_tag("capability:exogenous")
            or self.residual_forecaster.get_tag("capability:exogenous")
        )

        # The treatment_forecaster and outcome_forecaster must always support
        # in-sample predictions, thus forecaster's in-sample capability
        # depends only on residual_forecaster's in-sample capability
        in_sample = self.residual_forecaster.get_tag("capability:insample")

        # Use residual_forecaster to determine predictive capabilities
        pred_int = self.residual_forecaster.get_tag("capability:pred_int")
        pred_int_insample = self.residual_forecaster.get_tag(
            "capability:pred_int:insample"
        )

        # All components must handle missing and categorical data
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

        # Combine and set final capability tags
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
        """Fit DoubleMLForecaster on training data.

        1. Split X into X_exposure and X_confounder parts.
        2. Fit outcome forecaster on (y | confounders),
           and compute residuals y_res.
        3. Fit treatment forecaster on (exposures | confounders),
           and compute residuals X_exposure_res.
        4. Fit residual forecaster on (y_res | X_exposure_res)
           to estimate causal effect.
        5. Refit nuisance (outcome/treatment) forecasters with given `fh`.
        """
        # 1. Split X into exposure and confounder parts
        X_exposure, X_confounder = self._split_exogenous_data(X)

        # get in-sample forecasting horizon for residual computation
        if isinstance(y.index, pd.MultiIndex):
            time_idx = y.index.get_level_values(-1).unique()
        else:
            time_idx = y.index
        insample_fh = ForecastingHorizon(time_idx, is_relative=False)

        # 2. Fit outcome forecaster on confounders and compute residuals
        outcome_forecaster_insample = clone(self.outcome_forecaster)
        outcome_forecaster_insample.fit(y=y, X=X_confounder, fh=insample_fh)
        y_pred = outcome_forecaster_insample.predict(X=X_confounder)
        y_res = y - y_pred

        # 3. Fit treatment forecaster on confounders and compute exposure residuals
        if X_exposure is None:
            X_exposure_res = None
        else:
            treatment_forecaster_insample = clone(self.treatment_forecaster)
            treatment_forecaster_insample.fit(
                y=X_exposure, X=X_confounder, fh=insample_fh
            )
            X_exposure_pred = treatment_forecaster_insample.predict(X=X_confounder)
            X_exposure_res = X_exposure - X_exposure_pred

        # 4. Fit residual forecaster on residualized outcome and exposure
        self.residual_forecaster_ = clone(self.residual_forecaster)
        self.residual_forecaster_.fit(y=y_res, X=X_exposure_res, fh=fh)

        # 5. Refit nuisance models for forecasting
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
        """Generate forecasts using fitted DoubleMLForecaster.

        1. Split X into X_exposure and X_confounder parts.
        2. Compute residualized exposures using the treatment forecaster.
        3. Obtain base (confounder-driven) predictions from outcome forecaster.
        4. Obtain residual (causal) predictions from the residual forecaster.
        5. Combine both parts: final prediction = base + residual component.
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
            "exposure_vars": [0, "foo"],  # expected column names in test-suites
        }

        # More complex test parameters
        params2 = {
            "outcome_forecaster": NaiveForecaster(strategy="last"),
            "treatment_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": make_reduction(
                RandomForestRegressor(n_estimators=5, random_state=42),
                window_length=3,
            ),
            "exposure_vars": [0, "foo"],  # expected column names in test-suites
        }

        return [params1, params2]
