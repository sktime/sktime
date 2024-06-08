import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_random_state

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster


class EnbPIForecaster(BaseForecaster):
    """
    Ensemble Bootstrap Prediction Interval Forecaster.

    The forecaster combines sktime forecasters, with tsbootratp bootsrappers
    and the EnbPI algorithm implemented in fortuna.

    The forecaster is based upon the bagging forecaster and performs
    internally the following steps:
    For training:
        1. Fit a forecaster to each bootstrap sample
        2. Predict on the holdout set using each fitted forecaster

    For Prediction:
        1. Average the predictions of each fitted forecaster

    For Probabilistic Forecasting:
        1. Average the prediction of each fitted forecaster
        2. Use the EnbPI algorithm to calculate the prediction intervals


    Parameters
    ----------
    forecaster : estimator
        The base forecaster to fit to each bootstrap sample.
    bootstrap_transformer : tsbootstrap.BootstrapTransformer
        The transformer to fit to the target series to generate bootstrap samples.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    References
    ----------
    .. [1] Chen Xu & Yao Xie (2021). Conformal Prediction Interval for Dynamic
    Time-Series.
    """

    _tags = {
        "authors": ["benheid"],
        "python_dependencies": ["tsbootstrap>=0.1.0", "fortuna", "jax"],
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

    def __init__(self, forecaster=None, bootstrap_transformer=None, random_state=None):
        super().__init__()
        self.forecaster = forecaster
        self.forecaster_ = (
            clone(forecaster) if forecaster is not None else NaiveForecaster()
        )
        from tsbootstrap import MovingBlockBootstrap

        self.bootstrap_transformer = bootstrap_transformer
        self.bootstrap_transformer_ = (
            clone(bootstrap_transformer)
            if bootstrap_transformer is not None
            else MovingBlockBootstrap()
        )
        self.random_state = random_state

    def _fit(self, X, y, fh=None):
        self._fh = fh
        self._y_ix_names = y.index.names

        # random state handling passed into input estimators
        self.random_state_ = check_random_state(self.random_state)

        # fit/transform the transformer to obtain bootstrap samples
        bs_ts_index = list(
            self.bootstrap_transformer_.bootstrap(y, test_ratio=0, return_indices=True)
        )
        self.indexes = np.stack(list(map(lambda x: x[1], bs_ts_index)))
        bootstapped_ts = list(map(lambda x: x[0], bs_ts_index))

        self.forecasters = []
        self.preds = []
        # Fit Models per Bootstrap Sample
        for bs_ts in bootstapped_ts:
            bs_df = pd.DataFrame(bs_ts, index=y.index)
            forecaster = clone(self.forecaster_)
            forecaster.fit(y=bs_df, fh=fh, X=X)
            self.forecasters.append(forecaster)
            prediction = forecaster.predict(fh=y.index, X=X)
            self.preds.append(prediction)

        return self

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
        from fortuna.conformal import EnbPI

        preds = []
        for forecaster in self.forecasters:
            preds.append(forecaster.predict(fh=fh, X=X))

        intervals = []
        for cov in coverage:
            conformal_intervals = EnbPI().conformal_interval(
                bootstrap_indices=self.indexes,
                bootstrap_train_preds=np.stack(self.preds),
                bootstrap_test_preds=np.stack(preds),
                train_targets=self._y,
                error=1 - cov,
            )
            intervals.append(conformal_intervals)

        cols = pd.MultiIndex.from_product(
            [self._y.columns, coverage, ["lower", "upper"]]
        )
        fh_absolute = fh.to_absolute(self.cutoff)
        fh_absolute_idx = fh_absolute.to_pandas()
        pred_int = pd.DataFrame(
            np.concatenate(intervals, axis=1), index=fh_absolute_idx, columns=cols
        )
        return pred_int

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

        from sktime.utils.validation._dependencies import _check_soft_dependencies

        deps = cls.get_class_tag("python_dependencies")

        if _check_soft_dependencies(deps, severity="none"):
            from tsbootstrap import BlockResidualBootstrap

            params = [
                {},
                {
                    "forecaster": NaiveForecaster(),
                    "bootstrap_transformer": BlockResidualBootstrap(),
                },
            ]
        else:
            params = {}

        return params
