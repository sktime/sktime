# -*- coding: utf-8 -*-
__author__ = ["Matthew Middlehurst"]
__all__ = ["RotationForest"]

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y, check_random_state

from sktime.base._base import _clone_estimator
from sktime.exceptions import NotFittedError
from sktime.utils.validation import check_n_jobs


class RotationForest(BaseEstimator):
    def __init__(
        self,
        n_estimators=200,
        min_group=3,
        max_group=3,
        remove_proportion=0.5,
        base_estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.n_classes = 0
        self.n_instances = 0
        self.n_atts = 0
        self.classes_ = []
        self.estimators_ = []

        self._base_estimator = base_estimator
        self._min = 0
        self._ptp = 0
        self._useful_atts = []
        self._pcas = []
        self._groups = []
        self._n_jobs = n_jobs
        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

        super(RotationForest, self).__init__()

    def fit(self, X, y):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "ContinuousIntervalTree is not a time series classifier. "
                "A 2d numpy array is required."
            )
        X, y = check_X_y(X, y)

        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances, self.n_atts = X.shape
        self.classes_ = np.unique(y)
        self.n_classes = self.classes_.shape[0]

        if self.base_estimator is None:
            self._base_estimator = DecisionTreeClassifier(criterion="entropy")

        # replace missing values with 0 and remove useless attributes
        X = np.nan_to_num(X, False, 0, 0, 0)
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]

        # normalise attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        X_cls_split = [X[np.where(y == i)] for i in self.classes_]

        fit = Parallel(n_jobs=self._n_jobs)(
            delayed(self._fit_estimator)(
                X,
                X_cls_split,
                y,
                i,
            )
            for i in range(self.n_estimators)
        )

        self.estimators_, self._pcas, self._groups = zip(*fit)

        self._is_fitted = True
        return self

    def predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "ContinuousIntervalTree is not a time series classifier. "
                "A 2d numpy array is required."
            )

        # replace missing values with 0 and remove useless attributes
        X = np.nan_to_num(X, False, 0, 0, 0)
        X = X[:, self._useful_atts]

        # normalise the data.
        X = (X - self._min) / self._ptp

        y_probas = Parallel(n_jobs=self._n_jobs)(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                self._pcas[i],
                self._groups[i],
            )
            for i in range(self.n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        return output

    def _fit_estimator(self, X, X_cls_split, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (idx + 1)
        rng = check_random_state(rs)

        groups = self.generate_groups(rng)
        pcas = []

        # construct the slices to fit the PCAs too.
        for group in groups:
            classes = rng.choice(
                range(self.n_classes),
                size=rng.randint(1, self.n_classes + 1),
                replace=False,
            )

            # randomly add the classes with the randomly selected attributes.
            X_t = np.zeros((0, len(group)))
            for cls_idx in classes:
                c = X_cls_split[cls_idx]
                X_t = np.concatenate((X_t, c[:, group]), axis=0)

            sample_ind = rng.choice(
                X_t.shape[0],
                int(X_t.shape[0] * self.remove_proportion),
                replace=False,
            )
            X_t = X_t[sample_ind]

            # try to fit the PCA if it fails, remake it, and add 10 random data instances.
            while True:
                # ignore err state on PCA because we account if it fails. TODO error and check
                with np.errstate(divide="ignore", invalid="ignore"):
                    pca = PCA().fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)

        # merge all the pca_transformed data into one instance and build a classifier on it.
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        tree = _clone_estimator(self._base_estimator, random_state=rs)
        tree.fit(X_t, y)

        return tree, pcas, groups

    def _predict_proba_for_estimator(self, X, clf, pcas, groups):
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        return clf.predict_proba(X_t)

    def generate_groups(self, rng):
        permutation = rng.permutation((np.arange(0, self.n_atts)))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self.n_atts:
            n = rng.randint(group_size_count.shape[0])
            group_size_count[n] += 1
            n_attributes += self.min_group + n
            n_groups += 1

        groups = []
        current_attribute = 0
        current_size = 0
        for i in range(0, n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1

            n = self.min_group + current_size
            groups.append(np.zeros(n, dtype=np.int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(permutation.shape[0])]
                current_attribute += 1

        return groups
