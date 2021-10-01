# -*- coding: utf-8 -*-
"""RotationForest vector classifier.

Rotation Forest, sktime implementation for continuous values only.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["RotationForest"]

import time

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
    """Rotation Forest Classifier.

    Implementation of the Rotation Forest classifier described in Rodriguez et al
    (2013). [1]_
    Intended as a benchmark for time series data and a base classifier for
    transformation based appraoches such as ShapeletTransformClassifier, this sktime
    implementation only works with continuous attributes.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_group : int, default=3
        The minimum size of a group.
    max_group : int, default=3
        The maximum size of a group.
    remove_proportion : float, default=0.5
        The proportion of cases to be removed.
    base_estimator : BaseEstimator or None, default="None"
        Base estimator for the ensemble. By default uses the sklearn
        DecisionTreeClassifier using entropy as a splitting measure.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_instances : int
        The number of train cases.
    n_atts : int
        The number of attributes in each train case.
    classes_ : list
        The classes labels.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    transformed_data : list of shape (n_estimators) of ndarray
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    See Also
    --------
    ShapeletTransformClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /weka/classifiers/meta/RotationForest.java>`_.

    References
    ----------
    .. [1] Rodriguez, Juan José, Ludmila I. Kuncheva, and Carlos J. Alonso. "Rotation
       forest: A new classifier ensemble method." IEEE transactions on pattern analysis
       and machine intelligence 28.10 (2006).

    .. [2] Bagnall, A., et al. "Is rotation forest the best classifier for problems
       with continuous features?." arXiv preprint arXiv:1809.06705 (2018).

    Examples
    --------
    >>> from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> X_train = from_nested_to_3d_numpy(X_train)
    >>> X_test = from_nested_to_3d_numpy(X_test)
    >>> clf = RotationForest(n_estimators=10)
    >>> clf.fit(X_train, y_train)
    RotationForest(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        n_estimators=200,
        min_group=3,
        max_group=3,
        remove_proportion=0.5,
        base_estimator=None,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.n_classes = 0
        self.n_instances = 0
        self.n_atts = 0
        self.classes_ = []
        self.estimators_ = []
        self.transformed_data = []

        self._n_estimators = n_estimators
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
        """Fit a forest of trees on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : ndarray of shape = [n_instances,n_attributes]
            The training input samples.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self : object
        """
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForest is not a time series classifier. "
                "A 2d numpy array is required."
            )
        X, y = check_X_y(X, y)

        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances, self.n_atts = X.shape
        self.classes_ = np.unique(y)
        self.n_classes = self.classes_.shape[0]

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

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

        while (
            train_time < time_limit
            and self._n_estimators < self.contract_max_n_estimators
        ):
            self._n_estimators = 0
            self.estimators_ = []
            self._pcas = []
            self._groups = []

            fit = Parallel(n_jobs=self._n_jobs)(
                delayed(self._fit_estimator)(
                    X,
                    X_cls_split,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            estimators, pcas, groups, transformed_data = zip(*fit)

            self.estimators_ += estimators
            self._pcas = pcas
            self._groups = groups
            self.transformed_data += transformed_data

            self._n_estimators += self._n_jobs
            train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self._n_jobs)(
                delayed(self._fit_estimator)(
                    X,
                    X_cls_split,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            self.estimators_, self._pcas, self._groups, self.transformed_data = zip(
                *fit
            )

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : ndarray of shape = [n_instances,n_attributes]

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : ndarray of shape = [n_instances,n_attributes]

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForest is not a time series classifier. "
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
            for i in range(self._n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self._n_estimators
        )
        return output

    def _get_train_probs(self, X, y):
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForest is not a time series classifier. "
                "A 2d numpy array is required."
            )

        n_instances, n_atts = X.shape

        if n_instances != self.n_instances or n_atts != self.n_atts:
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        p = Parallel(n_jobs=self._n_jobs)(
            delayed(self._train_probas_for_estimator)(
                y,
                i,
            )
            for i in range(self._n_estimators)
        )
        y_probas, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(n_instances)
        for oob in oobs:
            for inst in oob:
                divisors[inst] += 1

        for i in range(n_instances):
            results[i] = (
                np.ones(self.n_classes) * (1 / self.n_classes)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes) * divisors[i])
            )

        return results

    def _fit_estimator(self, X, X_cls_split, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (idx + 1)
        rng = check_random_state(rs)

        groups = self._generate_groups(rng)
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
                # ignore err state on PCA because we account if it fails.
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

        return tree, pcas, groups, X_t if self.save_transformed_data else None

    def _predict_proba_for_estimator(self, X, clf, pcas, groups):
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        return clf.predict_proba(X_t)

    def _train_probas_for_estimator(self, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (idx + 1)
        rng = check_random_state(rs)

        indices = range(self.n_instances)
        subsample = rng.choice(self.n_instances, size=self.n_instances)
        oob = [n for n in indices if n not in subsample]

        clf = _clone_estimator(self._base_estimator, rs)
        clf.fit(self.transformed_data[idx][subsample], y[subsample])
        probas = clf.predict_proba(self.transformed_data[idx][oob])

        results = np.zeros((self.n_instances, self.n_classes))
        for n, proba in enumerate(probas):
            results[oob[n]] += proba

        return [results, oob]

    def _generate_groups(self, rng):
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
