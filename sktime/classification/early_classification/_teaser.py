# -*- coding: utf-8 -*-
"""TEASER early classifier.

An early classifier using a one class SVM's to determine decision safety with a
time series classifier.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["TEASER"]

import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.svm import OneClassSVM
from sklearn.utils import check_random_state

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import MUSE, WEASEL
from sktime.classification.feature_based import Catch22Classifier
from sktime.utils.validation.panel import check_X


class TEASER(BaseClassifier):
    """Two-tier Early and Accurate Series Classifier (TEASER).

    An early classifier which uses one class SVM's trained on prediction probabilities
    to determine whether an early prediction is safe or not.

    Overview:
        Build n classifiers, where n is the number of classification_points.
        For each classifier, train a one class svm used to determine prediction safety
        at that series length.
        Tune the number of consecutive safe svm predictions required to consider the
        prediction safe.

        While a prediction is still deemed unsafe:
            Make a prediction using the series length at classification point i.
            Decide whether the predcition is safe or not using decide_prediction_safety.

    Parameters
    ----------
    estimator: sktime classifier, default=None
        An sktime estimator to be built using the transformed data. Defaults to a
        WEASEL classifier.
    classification_points : List or None, default=None
        List of integer time series time stamps to build classifiers and allow
        predictions at. Early predictions must have a series length that matches a value
        in the _classification_points List. Duplicate values will be removed, and the
        full series length will be appeneded if not present.
        If None, will use 20 thresholds linearly spaces from 0 to the series length.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.

    Examples
    --------
    >>> from sktime.classification.early_classification import TEASER
    >>> from sktime.classification.interval_based import TimeSeriesForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = TEASER(
    ...     classification_points=[6, 16, 24],
    ...     estimator=TimeSeriesForestClassifier(n_estimators=10)
    ... )
    >>> clf.fit(X_train, y_train)
    TEASER(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:early_prediction": True,
    }

    def __init__(
        self,
        estimator=None,
        classification_points=None,
        n_jobs=1,
        random_state=None,
    ):
        self.estimator = estimator
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []
        self._svms = []
        self._consecutive_predictions = 0

        self._svm_gammas = [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1]
        self._svm_nu = 0.05
        self._svm_tol = 1e-4

        super(TEASER, self).__init__()

    def _fit(self, X, y):
        n_instances, _, series_length = X.shape

        self._classification_points = (
            copy.deepcopy(self.classification_points)
            if self.classification_points is not None
            else [round(series_length / i) for i in range(1, 21)]
        )
        # remove duplicates
        self._classification_points = list(set(self._classification_points))
        self._classification_points.sort()
        # remove classification points that are less than 3 time stamps
        self._classification_points = [i for i in self._classification_points if i >= 3]
        # make sure the full series length is included
        if self._classification_points[-1] != series_length:
            self._classification_points.append(series_length)
        # create dictionary of classification point indices
        self._classification_point_dictionary = {}
        for index, classification_point in enumerate(self._classification_points):
            self._classification_point_dictionary[classification_point] = index

        m = getattr(self.estimator, "n_jobs", None)
        threads = self._threads_to_use if m is None else 1

        fit = Parallel(n_jobs=threads)(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
            )
            for i in range(len(self._classification_points))
        )

        self._estimators, self._svms, X_svm, train_preds = zip(*fit)

        # tune consecutive predictions required to best harmonic mean
        best_hm = -1
        for g in range(2, min(6, len(self._classification_points))):
            # A List containing the state info for case, edited at each time stamp.
            # contains 1. the index of the time stamp, 2. the number of consecutive
            # positive decisions made, and 3. the prediction made
            state_info = [(0, 0, 0) for _ in range(n_instances)]
            # Stores whether we have made a final decision on a prediction, if true
            # state info wont be edited in later time stamps
            finished = [False for _ in range(n_instances)]

            for i in range(len(self._classification_points)):
                if i == len(self._classification_points) - 1:
                    decisions = [True for _ in range(n_instances)]
                elif self._svms[i] is not None:
                    decisions = self._svms[i].predict(X_svm[i]) == 1
                else:
                    decisions = [False for _ in range(n_instances)]

                # record consecutive class decisions
                state_info = [
                    (
                        # the classification point index
                        i,
                        # consecutive predictions, add one if positive decision and same
                        # class
                        state_info[n][1] + 1
                        if decisions[n] and train_preds[i][n] == state_info[i][2]
                        # set to 0 if the decision is negative, 1 if its positive but
                        # different class
                        else 1 if decisions[n] else 0,
                        # predicted class index
                        train_preds[i][n],
                    )
                    # if we have finished with this case do not edit the state info
                    if not finished[n] else state_info[n]
                    for n in range(n_instances)
                ]

                # safety decisions
                finished = [
                    True if state_info[n][1] >= g else False for n in range(n_instances)
                ]

            # calculate harmonic mean from finished state info
            accuracy = (
                np.sum(
                    [
                        1 if state_info[i][2] == self._class_dictionary[y[i]] else 0
                        for i in range(n_instances)
                    ]
                )
                / n_instances
            )
            earliness = (
                1
                - np.sum(
                    [
                        self.classification_points[state_info[i][0]] / series_length
                        for i in range(n_instances)
                    ]
                )
                / n_instances
            )
            hm = (2 * accuracy * earliness) / (accuracy + earliness)

            if hm > best_hm:
                best_hm = hm
                self._consecutive_predictions = g

        return self

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X):
        _, _, series_length = X.shape
        idx = self._classification_point_dictionary.get(series_length, -1)
        if idx == -1:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Current classification points: {self._classification_points}"
            )

        m = getattr(self._estimators[idx], "predict_proba", None)
        if callable(m):
            return self._estimators[idx].predict_proba(X)
        else:
            probas = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimators[idx].predict(X)
            for i in range(0, X.shape[0]):
                probas[i, self._class_dictionary[preds[i]]] = 1
            return probas

    def decide_prediction_safety(self, X, X_probabilities, state_info):
        """Decide on the safety of an early classification.

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series) of shape =
            [n_instances,n_dimensions,series_length] or pd.DataFrame with each column a
            dimension, each cell a pd.Series (any number of dimensions, equal or unequal
            length series).
            The prediction time series data.
        X_probabilities : 2D numpy array of shape = [n_instances,n_classes].
            The predicted probabilities for X.
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None for the
            first decision made, the returned List new_state_info for subsequent
            decisions.

        Returns
        -------
        decisions : List
            A List of booleans, containing the decision of whether a prediction is safe
            to use or not.
        new_state_info : List
            A List containing the state info for each decision in X, contains
            information for future decisions on the data.
        """
        X = check_X(X, coerce_to_numpy=True)

        n_instances, _, series_length = X.shape
        idx = self._classification_point_dictionary.get(series_length, -1)

        if idx == -1:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Current classification points: {self._classification_points}"
            )

        # If this is the smallest dataset, there should be no state_info, else we
        # should have state info for each, and they should all be the same length
        if state_info is None and (
            idx == 0 or idx == len(self._classification_points) - 1
        ):
            state_info = [(0, 0, 0) for _ in range(n_instances)]
        elif isinstance(state_info, list) and idx > 0:
            if not all(si[0] == idx for si in state_info):
                raise ValueError("All input instances must be of the same length.")
        else:
            raise ValueError(
                "state_info should be None for first time input, and a list of "
                "state_info outputs from the previous decision making for later inputs."
            )

        # if we have the full series, always return true
        if idx == len(self._classification_points) - 1:
            return [True for _ in range(n_instances)], None

        # find predicted class for each instance
        rng = check_random_state(self.random_state)
        preds = [
            int(rng.choice(np.flatnonzero(prob == prob.max())))
            for prob in X_probabilities
        ]

        # make a decision based on the one class svm prediction
        if self._svms[idx] is not None:
            X_svm = np.hstack((X_probabilities, np.ones((len(X), 1))))

            for i in range(len(X)):
                for n in range(self.n_classes_):
                    if n != preds[i]:
                        X_svm[i][self.n_classes_] = min(
                            X_svm[i][self.n_classes_], X_svm[i][preds[i]] - X_svm[i][n]
                        )

            decisions = self._svms[idx].predict(X_svm) == 1
        else:
            decisions = [False for _ in range(n_instances)]

        # record consecutive class decisions
        new_state_info = [
            (
                # next classification point index
                idx + 1,
                # consecutive predictions, add one if positive decision and same class
                state_info[i][1] + 1 if decisions[i] and preds[i] == state_info[i][2]
                # set to 0 if the decision is negative, 1 if its positive but different
                # class
                else 1 if decisions[i] else 0,
                # predicted class index
                preds[i],
            )
            for i in range(n_instances)
        ]

        # return the safety decisions and new state information for the instances
        return [
            True if new_state_info[i][1] >= self._consecutive_predictions else False
            for i in range(n_instances)
        ], new_state_info

    def _fit_estimator(self, X, y, i):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (i + 1)
        rng = check_random_state(rs)

        default = MUSE() if X.shape[1] > 1 else WEASEL()
        estimator = _clone_estimator(
            default if self.estimator is None else self.estimator,
            rng,
        )

        # fit estimator for this threshold
        estimator.fit(X[:, :, : self._classification_points[i]], y)

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            estimator.n_jobs = self._threads_to_use

        # get train set probability estimates for this estimator
        if callable(getattr(estimator, "_get_train_probs", None)):
            train_probs = estimator._get_train_probs(X, y)
        else:
            cv_size = 5
            _, counts = np.unique(y, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class

            train_probs = cross_val_predict(
                estimator, X, y=y, cv=cv_size, method="predict_proba"
            )

        rng = check_random_state(self.random_state)
        train_preds = [
            int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in train_probs
        ]
        train_probs = np.hstack((train_probs, np.ones((len(X), 1))))

        # create train set for svm using train probs with the minimum difference to the
        # predicted probability
        X_svm = []
        for i in range(len(X)):
            for n in range(self.n_classes_):
                if n != train_preds[i]:
                    train_probs[i][self.n_classes_] = min(
                        train_probs[i][self.n_classes_],
                        train_probs[i][train_preds[i]] - train_probs[i][n],
                    )

            if train_preds[i] == self._class_dictionary[y[i]]:
                X_svm.append(train_probs[i])

        cv_size = min(len(X_svm), 10)
        gs = GridSearchCV(
            OneClassSVM(tol=self._svm_tol, nu=self._svm_nu),
            {"gamma": self._svm_gammas},
            scoring="accuracy",
            cv=cv_size,
        )
        gs.fit(X_svm, np.ones(len(X_svm)))
        svm = gs.best_estimator_

        return estimator, svm, train_probs, train_preds

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        params = {
            "classification_points": [3],
            "estimator": Catch22Classifier(
                estimator=RandomForestClassifier(n_estimators=2)
            ),
        }
        return params
