# -*- coding: utf-8 -*-
from sktime.base import BaseEstimator

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import mlfinlab as ml


class Estimator(BaseEstimator):
    def __init__(
        self,
        estimator,
        param_grid,
        samples_col_name,
        labels_col_name,
        scoring="neg_log_loss",
        shuffle=False,
        test_size=0.25,
        n_splits=5,
        pct_embargo=0.01,
    ):
        self._samples_col_name = samples_col_name
        self._labels_col_name = labels_col_name
        self._n_splits = n_splits
        self._pct_embargo = pct_embargo
        self._shuffle = shuffle
        self._test_size = test_size
        self._estimator = estimator
        self._param_grid = param_grid
        self._scoring = scoring

    def fit(self, X, y, samples):
        train_idx, test_idx = train_test_split(
            np.arange(X.shape[0]), shuffle=self._shuffle, test_size=self._test_size
        )
        cv_gen = ml.cross_validation.PurgedKFold(
            samples_info_sets=samples[self._samples_col_name].iloc[train_idx],
            n_splits=self._n_splits,
            pct_embargo=self._pct_embargo,
        )
        gs = GridSearchCV(
            estimator=self._estimator,
            param_grid=self._param_grid,
            scoring=self._scoring,
            cv=cv_gen,
        )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx][self._labels_col_name]

        trained_estimator = gs.fit(X_train, y_train)
        self._fit_result = trained_estimator

        return self

    def predict(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
