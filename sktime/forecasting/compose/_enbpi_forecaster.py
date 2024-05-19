import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_random_state

from sktime.datatypes import convert
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import temporal_train_test_split
from sktime.transformations.bootstrap import MovingBlockBootstrapTransformer


class EnbPIForecaster(BaseForecaster):
    """
    Ensemble Bootstrap Prediction Interval Forecaster.

    The forecaster is inspired by the EnbPI algorithm proposed in [1].
    The forecaster is based upon the bagging forecaster and performs
    internally the following steps:
    For training:
        1. Fit a forecaster to each bootstrap sample
        2. Predict on the holdout set using each fitted forecaster
        3. Calculate the residuals using leave-one-out ensembles


    For Prediction:
        1. Average the predictions of each fitted forecaster

    For Probabilistic Forecasting:
        1. Average the prediction of each fitted forecaster
        2. Calculate conformal intervals using the residuals


    Parameters
    ----------
    forecaster : estimator
        The base forecaster to fit to each bootstrap sample.
    bootstrap_transformer : transformer
        The transformer to fit to the target series to generate bootstrap samples.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    method : str, default="conformal"
        The method to use for calculating prediction intervals. Options are:
        - "conformal": Use the conformal prediction intervals
        - "conformal_bonferroni": Use the conformal prediction intervals with
            Bonferroni correction
        - "empirical_residual": Use the empirical prediction intervals
        - "empirical": Use the empirical prediction intervals

    References
    ----------
    .. [1] Chen Xu & Yao Xie (2021). Conformal Prediction Interval for Dynamic
    Time-Series.
    """

    _tags = {
        "authors": ["benheid"],
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.DataFrame",
        # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",
        # which types do _fit, _predict, assume for X?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "requires-fh-in-fit": True,  # like AutoETS overwritten if forecaster not None
        "enforce_index_type": None,  # like AutoETS overwritten if forecaster not None
        "capability:insample": False,  # can the estimator make in-sample predictions?
        "capability:pred_int": True,  # can the estimator produce prediction intervals?
        "capability:pred_int:insample": False,  # ... for in-sample horizons?
    }

    def __init__(
        self, forecaster, bootstrap_transformer, random_state=None, method="conformal"
    ):
        super(EnbPIForecaster, self).__init__()

        self.forecaster = forecaster
        self.forecaster_ = (
            clone(forecaster) if forecaster is not None else NaiveForecaster()
        )

        self.bootstrap_transformer = bootstrap_transformer
        self.bootstrap_transformer_ = (
            clone(bootstrap_transformer)
            if bootstrap_transformer is not None
            else MovingBlockBootstrapTransformer()
        )

        self.random_state = random_state
        self.random_state = random_state
        self.method = method

    def _fit(self, X, y, fh=None):
        self._fh = fh
        self._y_ix_names = y.index.names

        # random state handling passed into input estimators
        self.random_state_ = check_random_state(self.random_state)

        # Temporal Split
        if X is None:
            y_train, y_fit_cp = temporal_train_test_split(
                y, fh=list(fh.to_relative(self.cutoff))
            )
            x_fit_cp = None
        else:
            y_train, y_fit_cp, X_train, x_fit_cp = temporal_train_test_split(
                y, X=X, fh=list(fh.to_relative(self.cutoff))
            )

        # fit/transform the transformer to obtain bootstrap samples
        y_bootstraps = self.bootstrap_transformer_.fit_transform(X=y_train)
        self._y_bs_ix = y_bootstraps.index

        if X is not None:
            # generate replicates of exogenous data for bootstrap
            X_inner = self._gen_X_bootstraps(X_train)
        else:
            X_inner = None

        self.forecasters = []
        self.preds = []
        self.residuals = []
        # Fit Models per Bootstrap Sample
        for bootstrap_ix in y_bootstraps.index.levels[0]:
            y_bootstrap = y_bootstraps.loc[bootstrap_ix]
            if X is not None:
                X_bootstrap = X_inner.loc[bootstrap_ix]
            else:
                X_bootstrap = None

            forecaster = clone(self.forecaster_)
            forecaster.fit(y=y_bootstrap, fh=fh, X=X_bootstrap)
            self.forecasters.append(forecaster)
            prediction = forecaster.predict(
                fh=list(fh.to_relative(self.cutoff)), X=x_fit_cp
            )
            self.preds.append(prediction)

        # Calculate Residuals using Leave-One-Out Cross Validation
        for i in range(len(self.preds)):
            pred = np.stack(self.preds[:i] + self.preds[i + 1 :], axis=0).mean(axis=0)
            self.residuals.append(y_fit_cp - pred)

        return self

    def _gen_X_bootstraps(self, X):
        """Generate replicates of exogenous data for bootstrap.

        Accesses self._y_bs_ix to obtain the index of the bootstrapped time series.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous time series, non-hierarchical

        Returns
        -------
        X_bootstraps : pd.DataFrame
            Bootstrapped exogenous data
        """
        # Copied from BaggingForecaster -> Need to be refactored
        if X is None:
            return None

        y_bs_ix = self._y_bs_ix

        # bootstrap instance index ends up at level -2
        inst_ix = y_bs_ix.get_level_values(-2).unique()
        X_repl = [X] * len(inst_ix)
        X_bootstraps = pd.concat(X_repl, keys=inst_ix)
        return X_bootstraps

    def _predict(self, X, fh=None):
        # Calculate Prediction Intervals using Bootstrap Samples

        preds = []
        for forecaster in self.forecasters:
            preds.append(forecaster.predict(fh=fh, X=X))

        return pd.DataFrame(
            np.stack(preds, axis=0).mean(axis=0),
            index=list(fh.to_absolute(self.cutoff)),
            columns=self._y.columns,
        )

    def _predict_interval(self, fh, X, coverage):
        y_pred = self.predict(fh=fh, X=X)

        return self._predict_interval_series(
            fh=fh,
            coverage=coverage,
            y_pred=y_pred,
        )

    def _predict_interval_series(self, fh, coverage, y_pred):
        """Compute prediction intervals predict_interval for series scitype."""
        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_idx = fh_absolute.to_pandas()

        residuals_matrix = np.stack(self.residuals)

        ABS_RESIDUAL_BASED = ["conformal", "conformal_bonferroni", "empirical_residual"]

        var_names = self._get_varnames()

        cols = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(index=fh_absolute_idx, columns=cols)
        for i, fh_ind in enumerate(fh_absolute):
            resids = residuals_matrix[:, i]
            abs_resids = np.abs(resids)
            coverage2 = np.repeat(coverage, 2)
            if self.method == "empirical":
                quantiles = 0.5 + np.tile([-0.5, 0.5], len(coverage)) * coverage2
                pred_int_row = np.quantile(resids, quantiles)
            if self.method == "empirical_residual":
                quantiles = 0.5 - 0.5 * coverage2
                pred_int_row = np.quantile(abs_resids, quantiles)
            elif self.method == "conformal_bonferroni":
                alphas = 1 - coverage2
                quantiles = 1 - alphas / len(fh)
                pred_int_row = np.quantile(abs_resids, quantiles)
            elif self.method == "conformal":
                quantiles = coverage2
                pred_int_row = np.quantile(abs_resids, quantiles)

            pred_int.loc[fh_ind] = pred_int_row

        y_pred = convert(y_pred, from_type=self._y_mtype_last_seen, to_type="pd.Series")
        y_pred.index = fh_absolute_idx

        for col in cols:
            if self.method in ABS_RESIDUAL_BASED:
                sign = 1 - 2 * (col[2] == "lower")
            else:
                sign = 1
            pred_int[col] = y_pred + sign * pred_int[col]

        return pred_int.convert_dtypes()

    def _update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.array
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        self.fit(y=self._y, X=self._X, fh=self._fh)
        return self

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.bootstrap import MovingBlockBootstrapTransformer

        return [
            {
                "forecaster": NaiveForecaster(),
                "bootstrap_transformer": MovingBlockBootstrapTransformer(),
            },
            {
                "forecaster": NaiveForecaster(),
                "bootstrap_transformer": MovingBlockBootstrapTransformer(),
                "method": "empirical_residual",
            },
        ]
