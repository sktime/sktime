#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecasters for combining forecasts via stacking."""

__author__ = ["mloning", "fkiraly", "indinewton"]
__all__ = ["StackingForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.split import SingleWindowSplitter
from sktime.utils.validation import array_is_int
from sktime.utils.validation.forecasting import check_cv, check_regressor
from sktime.utils.warnings import warn


class StackingForecaster(_HeterogenousEnsembleForecaster):
    """StackingForecaster.

    Stacks two or more Forecasters and uses a meta-model (regressor) to infer
    the final predictions from the predictions of the given forecasters.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    regressor: sklearn-like regressor, optional, default=None.
        The regressor is used as a meta-model and trained with the predictions
        of the ensemble forecasters as exog data and with y as endog data. The
        length of the data is dependent to the given fh. If None, then
        a GradientBoostingRegressor(max_depth=5) is used.
        The regressor can also be a sklearn.Pipeline().
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    cv : temporal cross-validation splitter, optional, default=None
        The sktime splitter to use for generating validation predictions for
        training the meta-regressor. The splitter must generate strictly
        temporal train/test splits, where every test window is after the
        corresponding train window. Test windows must match the forecasting
        horizon passed to ``fit``. If None, a single temporal validation window
        matching ``fh`` is used, preserving the previous behavior.

    Attributes
    ----------
    regressor_ : sklearn-like regressor
        Fitted meta-model (regressor)

    Examples
    --------
    >>> from sktime.forecasting.compose import StackingForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [
    ...     ("trend", PolynomialTrendForecaster()),
    ...     ("naive", NaiveForecaster()),
    ... ]
    >>> forecaster = StackingForecaster(forecasters=forecasters)
    >>> forecaster.fit(y=y, fh=[1,2,3])
    StackingForecaster(...)
    >>> y_pred = forecaster.predict()

    Use ``cv`` to train the meta-regressor from multiple temporal validation
    folds instead of one validation window:

    >>> from sktime.split import ExpandingWindowSplitter
    >>> cv = ExpandingWindowSplitter(
    ...     fh=[1, 2, 3], initial_window=36, step_length=24
    ... )
    >>> forecaster = StackingForecaster(forecasters=forecasters, cv=cv)
    >>> forecaster.fit(y=y, fh=[1, 2, 3])
    StackingForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["mloning", "fkiraly", "indinewton"],
        "capability:exogenous": True,
        "requires-fh-in-fit": True,
        "capability:missing_values": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        "capability:multivariate": False,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "X-y-must-have-same-index": True,
        "tests:skip_by_name": ["test_predict_time_index_with_X"],
    }

    def __init__(
        self, forecasters, regressor=None, random_state=None, n_jobs=None, cv=None
    ):
        self.regressor = regressor
        self.random_state = random_state
        self.cv = cv

        super().__init__(forecasters=forecasters, n_jobs=n_jobs)

        self._anytagis_then_set("capability:exogenous", True, False, forecasters)
        self._anytagis_then_set("capability:missing_values", False, True, forecasters)
        self._anytagis_then_set("fit_is_empty", False, True, forecasters)

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        forecasters = [x[1] for x in self.forecasters_]
        self.regressor_ = check_regressor(
            regressor=self.regressor, random_state=self.random_state
        )

        inner_fh = fh.to_relative(self.cutoff)
        cv = self._check_cv(inner_fh)
        X_meta, y_meta = self._get_meta_data(forecasters, y, X, inner_fh, cv)

        # fit final regressor on on validation window
        self.regressor_.fit(X_meta, y_meta)

        # refit forecasters on entire training series
        self._fit_forecasters(forecasters, y, fh=fh, X=X)

        return self

    def _check_cv(self, fh):
        """Validate or construct the temporal splitter for meta-training."""
        if self.cv is None:
            return SingleWindowSplitter(fh=fh)

        cv = check_cv(self.cv)
        split_type = cv.get_tag("split_type", "temporal", raise_error=False)
        if split_type != "temporal":
            raise ValueError(
                "`cv` must be an sktime temporal splitter, but found splitter "
                f"with split_type={split_type!r}."
            )
        return cv

    def _get_meta_data(self, forecasters, y, X, fh, cv):
        """Return meta-features and target values from temporal validation splits."""
        X_meta = []
        y_meta = []

        for fold, train_window, test_window in self._iter_cv_splits(cv, y, fh):
            y_train, y_test, X_train, X_test = self._get_fold_data(
                y, X, train_window, test_window
            )
            fold_fh = self._get_fold_fh(
                y_train, y_test, train_window, test_window, fh, fold
            )
            X_fold, y_fold = self._get_fold_meta_data(
                forecasters, y_train, y_test, X_train, X_test, fold_fh, fold
            )

            X_meta.append(X_fold)
            y_meta.append(y_fold)

        return self._concatenate_meta_data(X_meta, y_meta)

    def _iter_cv_splits(self, cv, y, fh):
        """Yield normalized and validated cv split windows."""
        splits = list(cv.split(y))

        if len(splits) == 0:
            raise ValueError("`cv` does not produce any valid train/test splits.")

        for fold, (train_window, test_window) in enumerate(splits):
            train_window, test_window = self._coerce_cv_split_windows(
                train_window, test_window, fold
            )
            self._check_cv_split(train_window, test_window, len(y), fold)
            self._check_fold_test_window_length(test_window, fh, fold)

            yield fold, train_window, test_window

    @staticmethod
    def _coerce_cv_split_windows(train_window, test_window, fold):
        """Return train and test windows as one-dimensional integer arrays."""
        train_window = np.asarray(train_window)
        test_window = np.asarray(test_window)

        if train_window.ndim != 1 or test_window.ndim != 1:
            raise ValueError(
                "`cv` must produce one-dimensional train/test windows, but "
                f"fold {fold} produced shapes {train_window.shape} and "
                f"{test_window.shape}."
            )

        for name, window in [("train", train_window), ("test", test_window)]:
            if len(window) > 0 and not np.issubdtype(window.dtype, np.integer):
                raise ValueError(
                    "`cv` must produce positional integer indices, but "
                    f"the {name} window in fold {fold} has dtype "
                    f"{window.dtype}."
                )

        return train_window.astype(int, copy=False), test_window.astype(int, copy=False)

    @staticmethod
    def _get_fold_data(y, X, train_window, test_window):
        """Return y and X slices for one fold."""
        y_train = y.iloc[train_window]
        y_test = y.iloc[test_window]

        if X is None:
            return y_train, y_test, None, None

        X_train = X.iloc[train_window]
        X_test = X.iloc[test_window]

        return y_train, y_test, X_train, X_test

    def _get_fold_meta_data(
        self, forecasters, y_train, y_test, X_train, X_test, fold_fh, fold
    ):
        """Return meta-features and targets for one temporal validation fold."""
        self._fit_forecasters(forecasters, y_train, fh=fold_fh, X=X_train)
        y_preds = self._predict_forecasters(fh=fold_fh, X=X_test)
        y_pred_arrays = self._check_fold_predictions(y_preds, y_test, fold)

        X_fold = np.column_stack(y_pred_arrays)
        y_fold = y_test.values
        self._check_fold_meta_data_shape(X_fold, y_fold, y_test, fold)

        return X_fold, y_fold

    @staticmethod
    def _check_cv_split(train_window, test_window, n_timepoints, fold):
        """Validate that a split is non-empty and temporally valid."""
        if len(train_window) == 0 or len(test_window) == 0:
            raise ValueError(
                f"`cv` produced an empty train or test window in fold {fold}."
            )

        StackingForecaster._check_window_is_strictly_increasing(
            train_window, "train", fold
        )
        StackingForecaster._check_window_is_strictly_increasing(
            test_window, "test", fold
        )
        StackingForecaster._check_window_bounds(
            train_window, n_timepoints, "train", fold
        )
        StackingForecaster._check_window_bounds(test_window, n_timepoints, "test", fold)

        if np.max(train_window) >= np.min(test_window):
            raise ValueError(
                "`cv` must produce temporal validation splits where all test "
                f"indices are after all train indices, but fold {fold} violates "
                "this requirement."
            )

    @staticmethod
    def _check_window_is_strictly_increasing(window, window_name, fold):
        """Validate that a split window is sorted and duplicate-free."""
        if len(window) > 1 and np.any(np.diff(window) <= 0):
            raise ValueError(
                "`cv` must produce strictly increasing, duplicate-free "
                f"{window_name} windows, but fold {fold} produced "
                f"{window.tolist()}."
            )

    @staticmethod
    def _check_window_bounds(window, n_timepoints, window_name, fold):
        """Validate that a split window contains valid positional indices."""
        if np.min(window) < 0 or np.max(window) >= n_timepoints:
            raise ValueError(
                "`cv` must produce positional indices within the bounds of `y`, "
                f"but the {window_name} window in fold {fold} contains "
                f"{window.tolist()} for y of length {n_timepoints}."
            )

    @staticmethod
    def _check_fold_test_window_length(test_window, fh, fold):
        """Validate that the validation window length matches the fit horizon."""
        expected = len(fh)
        actual = len(test_window)

        if actual != expected:
            raise ValueError(
                "`cv` forecasting horizon is incompatible with the forecasting "
                f"horizon passed to `fit` in fold {fold}. Expected {expected} "
                f"validation observations from fh={fh.to_numpy()}, but found "
                f"{actual}."
            )

    @staticmethod
    def _get_fold_fh(y_train, y_test, train_window, test_window, fh, fold):
        """Return fold forecasting horizon and check it is compatible with ``fh``."""
        if array_is_int(fh):
            fold_fh = ForecastingHorizon(
                test_window - train_window[-1], is_relative=True
            )
        else:
            fold_cutoff = pd.Index([y_train.index[-1]])
            fold_fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
                fold_cutoff
            )

        if not np.array_equal(fold_fh.to_numpy(), fh.to_numpy()):
            raise ValueError(
                "`cv` forecasting horizon is incompatible with the forecasting "
                f"horizon passed to `fit` in fold {fold}. Expected "
                f"{fh.to_numpy()}, but found {fold_fh.to_numpy()} for "
                f"train cutoff {y_train.index[-1]!r} and test index "
                f"{y_test.index.tolist()}."
            )

        return fold_fh

    @staticmethod
    def _check_fold_predictions(y_preds, y_test, fold):
        """Validate base forecaster predictions for one validation fold."""
        y_pred_arrays = []

        for i, y_pred in enumerate(y_preds):
            y_pred_array = StackingForecaster._coerce_fold_prediction(y_pred, fold, i)

            if len(y_pred_array) != len(y_test):
                raise ValueError(
                    "Base forecaster prediction length is incompatible with "
                    f"`cv` fold {fold}. Forecaster at position {i} predicted "
                    f"{len(y_pred_array)} rows, but the validation window has "
                    f"{len(y_test)} rows."
                )

            y_pred_arrays.append(y_pred_array)

        return y_pred_arrays

    @staticmethod
    def _coerce_fold_prediction(y_pred, fold, forecaster_position):
        """Return one base prediction as a one-dimensional numpy array."""
        y_pred_array = np.asarray(y_pred)

        if y_pred_array.ndim != 1:
            raise ValueError(
                "Base forecaster prediction shape is incompatible with "
                f"`cv` fold {fold}. Forecaster at position "
                f"{forecaster_position} returned shape {y_pred_array.shape}; "
                "expected a one-dimensional point forecast."
            )

        return y_pred_array

    @staticmethod
    def _check_fold_meta_data_shape(X_fold, y_fold, y_test, fold):
        """Validate assembled meta-feature and target arrays for one fold."""
        if X_fold.ndim != 2:
            raise ValueError(
                f"`cv` fold {fold} produced meta-features with shape "
                f"{X_fold.shape}; expected a two-dimensional array."
            )

        if X_fold.shape[0] != len(y_test) or len(y_fold) != len(y_test):
            raise ValueError(
                f"`cv` fold {fold} produced inconsistent meta data shapes: "
                f"X has {X_fold.shape[0]} rows, y has {len(y_fold)} rows, "
                f"and the validation window has {len(y_test)} rows."
            )

    @staticmethod
    def _concatenate_meta_data(X_meta, y_meta):
        """Concatenate per-fold meta-features and targets."""
        X_meta = np.concatenate(X_meta)
        y_meta = np.concatenate(y_meta)

        if X_meta.shape[0] != len(y_meta):
            raise ValueError(
                "Temporal stacking produced inconsistent meta data: "
                f"X has {X_meta.shape[0]} rows, but y has {len(y_meta)} rows."
            )

        return X_meta, y_meta

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        if update_params:
            warn("Updating `final regressor is not implemented", obj=self)
        for forecaster in self._get_forecaster_list():
            forecaster.update(y, X, update_params=update_params)
        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_preds = np.column_stack(self._predict_forecasters(fh=fh, X=X))
        y_pred = self.regressor_.predict(y_preds)
        # index = y_preds.index
        index = self.fh.to_absolute_index(self.cutoff)
        return pd.Series(y_pred, index=index, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.naive import NaiveForecaster

        f1 = NaiveForecaster()
        f2 = NaiveForecaster(strategy="mean", window_length=3)
        f3 = NaiveForecaster(strategy="last")
        f4 = NaiveForecaster(strategy="mean", window_length=2)
        params = [
            {"forecasters": [("f1", f1), ("f2", f2)]},
            {
                "forecasters": [("f3", f3), ("f4", f4)],
                "regressor": LinearRegression(),
            },
        ]

        return params
