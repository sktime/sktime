"""A rotation forest (RotF) vector classifier.

A rotation Forest sktime implementation for continuous values only. Fits sklearn
conventions.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["RotationForest"]

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from sktime.base._base import _clone_estimator
from sktime.utils.validation import check_n_jobs


class RotationForest(ClassifierMixin, BaseEstimator):
    """A rotation forest (RotF) vector classifier.

    Implementation of the Rotation Forest classifier described in Rodriguez et al
    (2013) [1]. Builds a forest of trees build on random portions of the data
    transformed using PCA.

    Intended as a benchmark for time series data and a base classifier for
    transformation based approaches such as ShapeletTransformClassifier, this sktime
    implementation only works with continuous attributes.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_group : int, default=3
        The minimum size of an attribute subsample group.
    max_group : int, default=3
        The maximum size of an attribute subsample group.
    remove_proportion : float, default=0.5
        The proportion of cases to be removed per group.
    base_estimator : BaseEstimator or None, default="None"
        Base estimator for the ensemble. By default, uses the sklearn
        ``DecisionTreeClassifier`` using entropy as a splitting measure.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding ``n_estimators``.
        Default of ``0`` means ``n_estimators`` is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators to build when ``time_limit_in_minutes`` is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit in ``transformed_data_`` for use in
        ``_get_train_probs``.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If ``int``, random_state is the seed used by the random number generator;
        If ``RandomState`` instance, random_state is the random number generator;
        If ``None``, the random number generator is the ``RandomState`` instance used
        by ``np.random``.

    Attributes
    ----------
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_instances_ : int
        The number of train cases in the training set.
    n_atts_ : int
        The number of attributes in the training set.
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed training dataset for all classifiers. Only saved when
        ``save_transformed_data`` is ``True``.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.

    See Also
    --------
    ShapeletTransformClassifier: A shapelet-based classifier using Rotation Forest.

    Notes
    -----
    For the Java version, see
    `tsml <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
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
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sktime.classification.sklearn import RotationForest
    >>>
    >>> X, y = load_iris(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
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

        super().__init__()

        if self.base_estimator is None:
            self._base_estimator = DecisionTreeClassifier(criterion="entropy")
        else:
            self._base_estimator = self.base_estimator

    def fit(self, X, y):
        """Fit a forest of trees on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_instances, n_attributes]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        from joblib import Parallel, delayed

        X, y = self._validate_data(
            X,
            y,
            dtype=[np.float32, np.float64],
            ensure_2d=True,
            allow_nd=True,
            force_all_finite=True,
        )
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances_, self.n_atts_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        # escape if only one class seen
        if self.n_classes_ == 1:
            self._is_fitted = True
            return self

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        # remove useless attributes
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]

        self._n_atts = X.shape[1]

        # normalise attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        X_cls_split = [X[np.where(y == i)] for i in self.classes_]

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self._pcas = []
            self._groups = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs)(
                    delayed(self._fit_estimator)(
                        X,
                        X_cls_split,
                        y,
                        i,
                    )
                    for i in range(self._n_jobs)
                )

                estimators, pcas, groups, transformed_data = zip(*fit)

                self.estimators_ += estimators
                self._pcas += pcas
                self._groups += groups
                self.transformed_data_ += transformed_data

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time
        else:
            self._n_estimators = self.n_estimators

            fit = Parallel(n_jobs=self._n_jobs)(
                delayed(self._fit_estimator)(
                    X,
                    X_cls_split,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            self.estimators_, self._pcas, self._groups, self.transformed_data_ = zip(
                *fit
            )

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_instances, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        check_is_fitted(self)
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
        X : 2d ndarray or DataFrame of shape = [n_instances, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        from joblib import Parallel, delayed

        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(
            X,
            dtype=[np.float32, np.float64],
            reset=False,
            ensure_2d=True,
            allow_nd=True,
            force_all_finite=True,
        )

        # replace missing values with 0 and remove useless attributes
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
            np.ones(self.n_classes_) * self._n_estimators
        )
        return output

    def _get_train_probs(self, X, y):
        from joblib import Parallel, delayed

        check_is_fitted(self)
        X = self._validate_data(
            X,
            dtype=[np.float32, np.float64],
            reset=False,
            ensure_2d=True,
            allow_nd=True,
            force_all_finite=True,
        )
        X = self._validate_data(X=X, reset=False)

        # handle the single-class-label case
        if len(self._class_dictionary) == 1:
            return np.repeat([[1]], len(X), axis=0)

        n_instances, n_atts = X.shape

        if n_instances != self.n_instances_ or n_atts != self.n_atts_:
            raise ValueError(
                "n_instances, n_atts mismatch. X should be the same as the training "
                "data used in fit for generating train probabilities."
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
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _fit_estimator(self, X, X_cls_split, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        groups = self._generate_groups(rng)
        pcas = []

        # construct the slices to fit the PCAs too.
        for group in groups:
            classes = rng.choice(
                range(self.n_classes_),
                size=rng.randint(1, self.n_classes_ + 1),
                replace=False,
            )

            # randomly add the classes with the randomly selected attributes.
            X_t = np.zeros((0, len(group)))
            for cls_idx in classes:
                c = X_cls_split[cls_idx]
                X_t = np.concatenate((X_t, c[:, group]), axis=0)

            sample_ind = rng.choice(
                X_t.shape[0],
                max(1, int(X_t.shape[0] * self.remove_proportion)),
                replace=False,
            )
            X_t = X_t[sample_ind]

            # try to fit the PCA if it fails, remake it, and add 10 random data
            # instances.
            while True:
                # ignore err state on PCA because we account if it fails.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # differences between os occasionally. seems to happen when there
                    # are low amounts of cases in the fit
                    pca = PCA(random_state=rs).fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)

        # merge all the pca_transformed data into one instance and build a classifier
        # on it.
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)
        tree = _clone_estimator(self._base_estimator, random_state=rs)
        tree.fit(X_t, y)

        return tree, pcas, groups, X_t if self.save_transformed_data else None

    def _predict_proba_for_estimator(self, X, clf, pcas, groups):
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        X_t = np.nan_to_num(X_t, False, 0, 0, 0)
        probas = clf.predict_proba(X_t)

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        return probas

    def _train_probas_for_estimator(self, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        indices = range(self.n_instances_)
        subsample = rng.choice(self.n_instances_, size=self.n_instances_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros((self.n_instances_, self.n_classes_))
        if len(oob) == 0:
            return [results, oob]

        clf = _clone_estimator(self._base_estimator, rs)
        clf.fit(self.transformed_data_[idx][subsample], y[subsample])
        probas = clf.predict_proba(self.transformed_data_[idx][oob])

        if probas.shape[1] != self.n_classes_:
            new_probas = np.zeros((probas.shape[0], self.n_classes_))
            for i, cls in enumerate(clf.classes_):
                cls_idx = self._class_dictionary[cls]
                new_probas[:, cls_idx] = probas[:, i]
            probas = new_probas

        for n, proba in enumerate(probas):
            results[oob[n]] += proba

        return [results, oob]

    def _generate_groups(self, rng):
        permutation = rng.permutation(np.arange(0, self._n_atts))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self._n_atts:
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
            groups.append(np.zeros(n, dtype=int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(permutation.shape[0])]
                current_attribute += 1

        return groups
