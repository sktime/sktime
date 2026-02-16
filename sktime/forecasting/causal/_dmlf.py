# copyright: sktime developers, BSD-3-Clause License
"""Implementation of Double Machine Learning methodology for forecasting."""

__all__ = ["DoubleMLForecaster"]
__author__ = ["geetu040", "XAheli"]

import math

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class DoubleMLForecaster(BaseForecaster):
    """Double Machine Learning forecaster for causal time-series forecasting.

    Implements an adaptation of Double Machine Learning (DML) framework [1]_
    for time-series, enabling deconfounded estimation of causal effects.

    The forecaster uses a three-step residualization process to separate
    causal effects from confounding influences:

    **Fit procedure**

    1. Split ``X`` into exposure variables ``X_exposure`` and
       confounder variables ``X_confounder``.
       ``X_exposure = X[exposure_vars]``,
       ``X_confounder = X.drop(columns=exposure_vars)``.

    2. Fit the outcome forecaster to obtain residuals
       ``outcome_fcst.fit(y=y, X=X_confounder, fh=y.index)``,
       ``y_pred = outcome_fcst.predict(X=X_confounder)``,
       ``y_res = y - y_pred``.

    3. Fit the treatment forecaster to obtain exposure residuals
       ``treatment_fcst.fit(y=X_exposure, X=X_confounder, fh=y.index)``,
       ``X_exposure_pred = treatment_fcst.predict(X=X_confounder)``,
       ``X_exposure_res = X_exposure - X_exposure_pred``.

    4. Fit the residual forecaster to learn deconfounded causal relationship.
       ``residual_fcst.fit(y=y_res, X=X_exposure_res, fh=fh)``.

    5. Refit the outcome and treatment forecasters for use during prediction.
       ``outcome_fcst.fit(y=y, X=X_confounder, fh=fh)``,
       ``treatment_fcst.fit(y=X_exposure, X=X_confounder, fh=fh)``.

    **Predict procedure**

    1. Split new ``X`` into exposure variables ``X_exposure`` and
       confounder variables ``X_confounder``.
       ``X_exposure = X[exposure_vars]``,
       ``X_confounder = X.drop(columns=exposure_vars)``.

    2. Compute the base (confounder-driven) forecast:
       ``y_pred_base = outcome_fcst.predict(X_confounder)``.

    3. Compute the residualized exposures:
       ``X_exposure_pred = treatment_fcst.predict(X_confounder)``.
       ``X_exposure_res = X_exposure - X_exposure_pred``.

    4. Compute the causal (residual) forecast:
       ``y_pred_res = residual_fcst.predict(X=X_exposure_res)``.

    5. Combine both components to obtain the final prediction:
       ``y_pred = y_pred_base + y_pred_res``.

    Parameters
    ----------
    outcome_fcst : sktime forecaster
        Base forecaster modeling the outcome variable conditional on
        confounders.

    treatment_fcst : sktime forecaster
        Forecaster modeling the exposure variables conditional on confounders.

    residual_fcst : sktime forecaster, optional (default=None)
        Forecaster modeling the residual (deconfounded) relationship between
        outcome and treatment. If not provided, a default forecaster is created
        using ``make_reduction(LinearRegression(), strategy="recursive")``,
        a recursive reduction forecaster built from a linear regression model,
        providing a simple and interpretable baseline.

    exposure_vars : list of str, optional (default=None)
        Names of columns in ``X`` representing exposure (treatment) variables.
        The remaining columns are treated as confounders. If ``None``, all
        features in ``X`` are treated as confounders and used only with the
        outcome forecaster. In this case, the treatment forecaster is not used,
        and ``None`` exposure residuals are passed to the residual forecaster,
        which therefore operates only on ``y``.

    starting_window : float, int or None, optional (default=0.1)
        If float, must be between 0.0 and 1.0, and is interpreted as the proportion
        of the starting dataset to ignore when doing in-sample predictions.
        Proportions are rounded to the next higher integer count of samples (ceil).
        If int, is interpreted as total number of samples to ignore.

    Attributes
    ----------
    outcome_fcst_ : sktime forecaster
        Fitted clone of the outcome forecaster.

    treatment_fcst_ : sktime forecaster
        Fitted clone of the treatment forecaster.

    residual_fcst_ : sktime forecaster
        Fitted clone of the residual forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.causal import DoubleMLForecaster
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
    >>> outcome_fcst = NaiveForecaster()
    >>> treatment_fcst = NaiveForecaster()
    >>>
    >>> # Create DoubleMLForecaster
    >>> dml_forecaster = DoubleMLForecaster(
    ...     outcome_fcst=outcome_fcst,
    ...     treatment_fcst=treatment_fcst,
    ...     exposure_vars=exposure_vars
    ... )
    >>>
    >>> # Fit and predict
    >>> fh = [1, 2, 3]
    >>> dml_forecaster.fit(y_train, X=X_train, fh=fh)
    DoubleMLForecaster(exposure_vars=['GNP'], outcome_fcst=NaiveForecaster(),
                       residual_fcst=RecursiveTabularRegressionForecaster(estimator=LinearRegression(),
                                                                          window_length=3),
                       treatment_fcst=NaiveForecaster())
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
      wrapped using a utility such as ``OosForecaster`` to enable this
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
        outcome_fcst,
        treatment_fcst,
        residual_fcst=None,
        exposure_vars=None,
        starting_window=0.1,
    ):
        self.outcome_fcst = outcome_fcst
        self.treatment_fcst = treatment_fcst
        self.residual_fcst = residual_fcst
        self.exposure_vars = exposure_vars
        self.starting_window = starting_window

        # fitted copies of user passed forecasters
        self.outcome_fcst_ = None
        self.treatment_fcst_ = None
        self.residual_fcst_ = None

        super().__init__()

        # Handle null residual_fcst
        if self.residual_fcst is None:
            from sktime.forecasting.compose import make_reduction

            self.residual_fcst = make_reduction(
                LinearRegression(),
                strategy="recursive",
                window_length=3,
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
            self.outcome_fcst.get_tag("capability:exogenous")
            or self.treatment_fcst.get_tag("capability:exogenous")
            or self.residual_fcst.get_tag("capability:exogenous")
        )

        # The treatment_fcst and outcome_fcst must always support
        # in-sample predictions, thus forecaster's in-sample capability
        # depends only on residual_fcst's in-sample capability
        in_sample = self.residual_fcst.get_tag("capability:insample")

        # Use residual_fcst to determine predictive capabilities
        pred_int = self.residual_fcst.get_tag("capability:pred_int")
        pred_int_insample = self.residual_fcst.get_tag("capability:pred_int:insample")

        # All components must handle missing/categorical data and fh
        miss = (
            self.outcome_fcst.get_tag("capability:missing_values")
            and self.treatment_fcst.get_tag("capability:missing_values")
            and self.residual_fcst.get_tag("capability:missing_values")
        )
        cat = (
            self.outcome_fcst.get_tag("capability:categorical_in_X")
            and self.treatment_fcst.get_tag("capability:categorical_in_X")
            and self.residual_fcst.get_tag("capability:categorical_in_X")
        )
        req_fh = (
            self.outcome_fcst.get_tag("requires-fh-in-fit")
            or self.treatment_fcst.get_tag("requires-fh-in-fit")
            or self.residual_fcst.get_tag("requires-fh-in-fit")
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
                "requires-fh-in-fit": req_fh,
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

        starting_window = self.starting_window
        if isinstance(starting_window, float):
            starting_window = math.ceil(self.starting_window * len(time_idx))

        time_idx = time_idx[starting_window:]
        time_idx_filter = y.index.get_level_values(-1).isin(time_idx)
        insample_fh = ForecastingHorizon(time_idx, is_relative=False)

        # 2. Fit outcome forecaster on confounders and compute residuals
        outcome_fcst_insample = clone(self.outcome_fcst)
        outcome_fcst_insample.fit(y=y, X=X_confounder, fh=insample_fh)
        y_pred = outcome_fcst_insample.predict(X=X_confounder)
        y_res = y[time_idx_filter] - y_pred

        # 3. Fit treatment forecaster on confounders and compute exposure residuals
        if X_exposure is None:
            X_exposure_res = None
        else:
            treatment_fcst_insample = clone(self.treatment_fcst)
            treatment_fcst_insample.fit(y=X_exposure, X=X_confounder, fh=insample_fh)
            X_exposure_pred = treatment_fcst_insample.predict(X=X_confounder)
            X_exposure_res = X_exposure[time_idx_filter] - X_exposure_pred

        # 4. Fit residual forecaster on residualized outcome and exposure
        self.residual_fcst_ = clone(self.residual_fcst)
        self.residual_fcst_.fit(y=y_res, X=X_exposure_res, fh=fh)

        # 5. Refit nuisance models for forecasting
        self.outcome_fcst_ = clone(self.outcome_fcst)
        self.outcome_fcst_.fit(y=y, X=X_confounder, fh=fh)

        if X_exposure is not None:
            self.treatment_fcst_ = clone(self.treatment_fcst)
            self.treatment_fcst_.fit(y=X_exposure, X=X_confounder, fh=fh)

    def _compute_X_exposure_res(self, X_exposure=None, X_confounder=None, fh=None):
        if X_exposure is None:
            return None

        X_exposure_pred = self.treatment_fcst_.predict(fh=fh, X=X_confounder)
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

        pred_base = self.outcome_fcst_.predict(fh=fh, X=X_confounder)
        pred_res = self.residual_fcst_.predict(fh=fh, X=X_exposure_res)

        return pred_base + pred_res

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Generate prediction intervals for DoubleMLForecaster."""
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_base = self.outcome_fcst_.predict(fh=fh, X=X_confounder)
        pred_int_res = self.residual_fcst_.predict_interval(
            fh=fh, X=X_exposure_res, coverage=coverage
        )

        return self._add_det_to_proba(pred_int_res, pred_base)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Generate quantile forecasts for DoubleMLForecaster."""
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_base = self.outcome_fcst_.predict(fh=fh, X=X_confounder)
        pred_quantiles_res = self.residual_fcst_.predict_quantiles(
            fh=fh, X=X_exposure_res, alpha=alpha
        )

        return self._add_det_to_proba(pred_quantiles_res, pred_base)

    def _predict_var(self, fh, X=None, cov=False):
        """Generate predictive variances for DoubleMLForecaster."""
        X_exposure, X_confounder = self._split_exogenous_data(X)

        X_exposure_res = self._compute_X_exposure_res(
            X_exposure=X_exposure, X_confounder=X_confounder, fh=fh
        )

        pred_var_res = self.residual_fcst_.predict_var(fh=fh, X=X_exposure_res, cov=cov)

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

        pred_base = self.outcome_fcst_.predict(fh=fh, X=X_confounder)
        pred_proba_res = self.residual_fcst_.predict_proba(
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
            "outcome_fcst": NaiveForecaster(strategy="last"),
            "treatment_fcst": NaiveForecaster(strategy="last"),
            "residual_fcst": make_reduction(LinearRegression(), window_length=3),
            "exposure_vars": [0, "foo"],  # expected column names in test-suites
        }

        # More complex test parameters
        params2 = {
            "outcome_fcst": NaiveForecaster(strategy="last"),
            "treatment_fcst": NaiveForecaster(strategy="last"),
            "residual_fcst": make_reduction(
                RandomForestRegressor(n_estimators=5, random_state=42),
                window_length=3,
            ),
            "exposure_vars": [0, "foo"],  # expected column names in test-suites
        }

        return [params1, params2]
