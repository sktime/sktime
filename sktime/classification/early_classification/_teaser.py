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
    one_class_classifier: one-class sklearn classifier, default=None
        An sklearn one-class classifier used to determine whether an early decision is
        safe. Defaults to a tuned one-class SVM classifier.
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
    return_safety_decisions : bool, default=True
        Whether to return decisions and decision state information alongside
        predictions/predicted probabiltiies in predict and predict_proba.

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
        one_class_classifier=None,
        classification_points=None,
        n_jobs=1,
        random_state=None,
        return_safety_decisions=True,
    ):
        self.estimator = estimator
        self.one_class_classifier = one_class_classifier
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.return_safety_decisions = return_safety_decisions

        self._estimators = []
        self._one_class_classifiers = []
        self._classification_points = []
        self._consecutive_predictions = 0

        self._svm_gammas = [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1]
        self._svm_nu = 0.05
        self._svm_tol = 1e-4

        super(TEASER, self).__init__()

    def _fit(self, X, y):
        m = getattr(self.estimator, "predict_proba", None)
        if not callable(m):
            raise ValueError("Base estimator must have a predict_proba method.")

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

        self._estimators, self._one_class_classifiers, X_oc, train_preds = zip(*fit)

        # tune consecutive predictions required to best harmonic mean
        best_hm = -1
        for g in range(2, min(6, len(self._classification_points))):
            state_info, _ = self._predict_n_timestamps(
                train_preds, X_oc, g, len(self._classification_points)
            )

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

    def predict(self, X, state_info=None):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None or an
            empty List for the first decision made, the returned List new_state_info for
            subsequent decisions.
            If no state_info is provided and the input series_length is greater than
            the first classification_points time stamp, all previous time stamps are
            considered up to the input series_length and the class value for the first
            safe prediction is returned.

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        decisions : List
            A List of booleans, containing the decision of whether a prediction is safe
            to use or not. Returned if return_safety_decisions is True.
        new_state_info : List
            A List containing the state info for each decision in X, contains
            information for future decisions on the data. Returned if
            return_safety_decisions is True.
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._predict(X, state_info=state_info)

    def _predict(self, X, state_info=None):
        out = self._predict_proba(X)
        probas = out[0] if self.return_safety_decisions else out

        rng = check_random_state(self.random_state)
        preds = np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probas
            ]
        )

        return (preds, out[1], out[2]) if self.return_safety_decisions else preds

    def predict_proba(self, X, state_info=None):
        """Decide on the safety of an early classification.

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb
        state_info : List or None
            A List containing the state info for each decision in X. contains
            information for future decisions on the data. Inputs should be None or an
            empty List for the first decision made, the returned List new_state_info for
            subsequent decisions.
            If no state_info is provided and the input series_length is greater than
            the first classification_points time stamp, all previous time stamps are
            considered up to the input series_length and the probabilities for the first
            safe prediction are returned.

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        decisions : List
            A List of booleans, containing the decision of whether a prediction is safe
            to use or not. Returned if return_safety_decisions is True.
        new_state_info : List
            A List containing the state info for each decision in X, contains
            information for future decisions on the data. Returned if
            return_safety_decisions is True.
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._predict_proba(X, state_info=state_info)

    def _predict_proba(self, X, state_info=None):
        n_instances, _, series_length = X.shape
        idx = self._classification_point_dictionary.get(series_length, -1)
        if idx == -1:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Current classification points: {self._classification_points}"
            )

        # if no state_info is provided, consider all previous time stamps up to the
        # input series_length
        # else use previous state_info if idx > 0
        if state_info is None and idx > 0:
            m = getattr(self.estimator, "n_jobs", None)
            threads = self._threads_to_use if m is None else 1

            out = Parallel(n_jobs=threads)(
                delayed(self._predict_proba_for_estimator)(
                    X,
                    i,
                )
                for i in range(len(idx))
            )

            X_oc, probas, preds = zip(*out)

            new_state_info, decisions = self._predict_n_timestamps(
                preds, X_oc, self._consecutive_predictions, idx + 1
            )

            probas = np.array([probas[new_state_info[i][0]][i] for i in n_instances])
        else:
            # if this is the smallest dataset, there should be no state_info, else we
            # should have state info for each, and they should all be the same length
            if idx == 0 and (state_info is None or state_info == []):
                state_info = [(0, 0, 0) for _ in range(n_instances)]
            elif isinstance(state_info, list) and idx > 0:
                if not all(si[0] == idx - 1 for si in state_info):
                    raise ValueError(
                        "All state_info input instances must be from the "
                        "previous classification point series length."
                    )
            else:
                raise ValueError(
                    "state_info should be None or an empty List for first time input, "
                    "and a list of state_info outputs from the previous decision "
                    "making for later inputs."
                )

            probas = self._estimators[idx].predict_proba(X)

            # find predicted class for each instance
            rng = check_random_state(self.random_state)
            preds = [
                int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in probas
            ]

            # make a decision based on the one class classifier prediction
            if self._one_class_classifiers[idx] is not None:
                X_oc = np.hstack((probas, np.ones((len(X), 1))))

                for i in range(len(X)):
                    for n in range(self.n_classes_):
                        if n != preds[i]:
                            X_oc[i][self.n_classes_] = min(
                                X_oc[i][self.n_classes_], X_oc[i][preds[i]] - X_oc[i][n]
                            )

                decisions = self._one_class_classifiers[idx].predict(X_oc) == 1
            else:
                decisions = [False for _ in range(n_instances)]

            # record consecutive class decisions
            new_state_info = [
                self._update_state_info(decisions, preds, state_info, i, idx)
                for i in range(n_instances)
            ]

            # if we have the full series, always decide True
            if idx == len(self._classification_points) - 1:
                decisions = [True for _ in range(n_instances)]
            else:
                decisions = [
                    True if state_info[i][1] >= self._consecutive_predictions else False
                    for i in range(n_instances)
                ]

        return (
            (probas, decisions, new_state_info)
            if self.return_safety_decisions
            else probas
        )

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
            train_probas = estimator._get_train_probs(X, y)
        else:
            cv_size = 5
            _, counts = np.unique(y, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class

            train_probas = cross_val_predict(
                estimator, X, y=y, cv=cv_size, method="predict_proba"
            )

        train_preds = [
            int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in train_probas
        ]
        train_probas = np.hstack((train_probas, np.ones((len(X), 1))))

        # create train set for the one class classifier using train probas with the
        # minimum difference to the predicted probability
        X_oc = []
        for i in range(len(X)):
            for n in range(self.n_classes_):
                if n != train_preds[i]:
                    train_probas[i][self.n_classes_] = min(
                        train_probas[i][self.n_classes_],
                        train_probas[i][train_preds[i]] - train_probas[i][n],
                    )

            if train_preds[i] == self._class_dictionary[y[i]]:
                X_oc.append(train_probas[i])

        one_class_classifier = None
        if len(X_oc) > 1:
            if self.one_class_classifier is None:
                cv_size = min(len(X_oc), 10)
                gs = GridSearchCV(
                    OneClassSVM(tol=self._svm_tol, nu=self._svm_nu),
                    {"gamma": self._svm_gammas},
                    scoring="accuracy",
                    cv=cv_size,
                )
                gs.fit(X_oc, np.ones(len(X_oc)))
                one_class_classifier = gs.best_estimator_
            else:
                one_class_classifier = _clone_estimator(
                    self.one_class_classifier, random_state=rs
                )
                one_class_classifier.fit(X_oc, np.ones(len(X_oc)))

        return estimator, one_class_classifier, train_probas, train_preds

    def _predict_proba_for_estimator(self, X, i):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (i + 1)
        rng = check_random_state(rs)

        probas = self._estimators[i].predict_proba(
            X[:, :, : self._classification_points[i]]
        )
        preds = [int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in probas]

        # create data set for the one class classifier using predicted probas with the
        # minimum difference to the predicted probability
        X_oc = []
        if self._one_class_classifiers[i] is not None:
            X_oc = np.hstack((probas, np.ones((len(X), 1))))

            for i in range(len(X)):
                for n in range(self.n_classes_):
                    if n != preds[i]:
                        X_oc[i][self.n_classes_] = min(
                            X_oc[i][self.n_classes_], X_oc[i][preds[i]] - X_oc[i][n]
                        )

        return X_oc, probas, preds

    def _predict_n_timestamps(self, preds, X_oc, consecutive_predictions, n_timestamps):
        n_instances = len(preds[0])

        # a List containing the state info for case, edited at each time stamp.
        # contains 1. the index of the time stamp, 2. the number of consecutive
        # positive decisions made, and 3. the prediction made
        state_info = [(0, 0, 0) for _ in range(n_instances)]
        # stores whether we have made a final decision on a prediction, if true
        # state info wont be edited in later time stamps
        finished = [False for _ in range(n_instances)]

        for i in range(n_timestamps):
            if i == len(self._classification_points) - 1:
                decisions = [True for _ in range(n_instances)]
            elif self._one_class_classifiers[i] is not None:
                decisions = self._one_class_classifiers[i].predict(X_oc[i]) == 1
            else:
                decisions = [False for _ in range(n_instances)]

            # record consecutive class decisions
            state_info = [
                self._update_state_info(decisions, preds[i], state_info, n, i)
                # if we have finished with this case do not edit the state info
                if not finished[n] else state_info[n]
                for n in range(n_instances)
            ]

            # safety decisions
            finished = [
                True if state_info[n][1] >= consecutive_predictions else False
                for n in range(n_instances)
            ]

        return state_info, finished

    @staticmethod
    def _update_state_info(decisions, preds, state_info, idx, time_stamp):
        # consecutive predictions, add one if positive decision and same class
        if decisions[idx] and preds[idx] == state_info[idx][2]:
            return time_stamp, state_info[idx][1] + 1, preds[idx]
        # set to 0 if the decision is negative, 1 if its positive but different class
        else:
            return time_stamp, 1 if decisions[idx] else 0, preds[idx]

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
            "return_safety_decisions": False,
        }
        return params
