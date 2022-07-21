# -*- coding: utf-8 -*-
"""Probability Threshold Early Classifier.

An early classifier using a prediction probability threshold with a time series
classifier.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ProbabilityThresholdEarlyClassifier"]

import copy
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

from sktime.base._base import _clone_estimator
from sktime.classification.early_classification import BaseEarlyClassifier
from sktime.classification.interval_based import DrCIF


class ProbabilityThresholdEarlyClassifier(BaseEarlyClassifier):
    """Probability Threshold Early Classifier.

    An early classifier which uses a threshold of prediction probability to determine
    whether an early prediction is safe or not.

    Overview:
        Build n classifiers, where n is the number of classification_points.
        While a prediction is still deemed unsafe:
            Make a prediction using the series length at classification point i.
            Decide whether the predcition is safe or not using decide_prediction_safety.

    Parameters
    ----------
    probability_threshold : float, default=0.85
        The class prediction probability required to deem a prediction as safe.
    consecutive_predictions : int, default=1
        The number of consecutive predictions for a class above the threshold required
        to deem a prediction as safe.
    estimator: sktime classifier, default=None
        An sktime estimator to be built using the transformed data. Defaults to a
        default DrCIF classifier.
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
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The full length of each series.
    classes_ : list
        The unique class labels.
    state_info : 2d np.ndarray (4 columns)
        Information stored about input instances after the decision-making process in
        update/predict methods. Used in update methods to make decisions based on
        the resutls of previous method calls.
        Records in order: the time stamp index, the number of consecutive decisions
        made, the predicted class and the series length.

    Examples
    --------
    >>> from sktime.classification.early_classification import (
    ...     ProbabilityThresholdEarlyClassifier
    ... )
    >>> from sktime.classification.interval_based import TimeSeriesForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = ProbabilityThresholdEarlyClassifier(
    ...     classification_points=[6, 16, 24],
    ...     estimator=TimeSeriesForestClassifier(n_estimators=10)
    ... )
    >>> clf.fit(X_train, y_train)
    ProbabilityThresholdEarlyClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        estimator=None,
        probability_threshold=0.85,
        consecutive_predictions=1,
        classification_points=None,
        n_jobs=1,
        random_state=None,
    ):
        self.estimator = estimator
        self.probability_threshold = probability_threshold
        self.consecutive_predictions = consecutive_predictions
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0

        super(ProbabilityThresholdEarlyClassifier, self).__init__()

    def _fit(self, X, y):
        m = getattr(self.estimator, "predict_proba", None)
        if self.estimator is not None and not callable(m):
            raise ValueError("Base estimator must have a predict_proba method.")

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._classification_points = (
            copy.deepcopy(self.classification_points)
            if self.classification_points is not None
            else [round(self.series_length_ / i) for i in range(1, 21)]
        )
        # remove duplicates
        self._classification_points = list(set(self._classification_points))
        self._classification_points.sort()
        # remove classification points that are less than 3 time stamps
        self._classification_points = [i for i in self._classification_points if i >= 3]
        # make sure the full series length is included
        if self._classification_points[-1] != self.series_length_:
            self._classification_points.append(self.series_length_)
        # create dictionary of classification point indices
        self._classification_point_dictionary = {}
        for index, classification_point in enumerate(self._classification_points):
            self._classification_point_dictionary[classification_point] = index

        m = getattr(self.estimator, "n_jobs", None)
        threads = self._threads_to_use if m is None else 1

        self._estimators = Parallel(n_jobs=threads)(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
            )
            for i in range(len(self._classification_points))
        )

        return self

    def _predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        out = self._predict_proba(X)
        return self._proba_output_to_preds(out)

    def _update_predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        out = self._update_predict_proba(X)
        return self._proba_output_to_preds(out)

    def _predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray]:
        n_instances, _, series_length = X.shape

        # maybe use the largest index that is smaller than the series length
        next_idx = self._get_next_idx(series_length) + 1

        # if the input series length is invalid
        if next_idx == 0:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Input series length must be greater then the first point. "
                f"Current classification points: {self._classification_points}"
            )

        m = getattr(self.estimator, "n_jobs", None)
        threads = self._threads_to_use if m is None else 1

        # compute all new updates since then
        out = Parallel(n_jobs=threads)(
            delayed(self._predict_proba_for_estimator)(
                X,
                i,
            )
            for i in range(0, next_idx)
        )
        probas, preds = zip(*out)

        # a List containing the state info for case, edited at each time stamp.
        # contains 1. the index of the time stamp, 2. the number of consecutive
        # positive decisions made, and 3. the prediction made
        self.state_info = np.zeros((len(preds[0]), 4), dtype=int)

        # only compute new indices
        for i in range(0, next_idx):
            accept_decision, self.state_info = self._decide_prediction_safety(
                i,
                probas[i],
                preds[i],
                self.state_info,
            )

        probas = np.array(
            [
                probas[self.state_info[i][0]][i]
                if accept_decision[i]
                else [-1 for _ in range(self.n_classes_)]
                for i in range(n_instances)
            ]
        )

        return probas, accept_decision

    def _update_predict_proba(self, X) -> Tuple[np.ndarray, np.ndarray]:
        n_instances, _, series_length = X.shape

        # maybe use the largest index that is smaller than the series length
        next_idx = self._get_next_idx(series_length) + 1

        # remove cases where a positive decision has been made
        state_info = self.state_info[
            self.state_info[:, 1] < self._consecutive_predictions
        ]

        # determine last index used
        last_idx = np.max(state_info[0][0]) + 1

        # if the input series length is invalid
        if next_idx == 0:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Input series length must be greater then the first point. "
                f"Current classification points: {self._classification_points}"
            )
        # check state info and X have the same length
        if len(X) > len(state_info):
            raise ValueError(
                f"Input number of instances does not match the length of recorded "
                f"state_info: {len(state_info)}. Cases with positive decisions "
                f"returned should be removed from the array with the row ordering "
                f"preserved, or the state information should be reset if new data is "
                f"used."
            )
        # check if series length has increased from last time
        elif last_idx >= next_idx:
            raise ValueError(
                f"All input instances must be from a larger classification point time "
                f"stamp than the recorded state information. Required series length "
                f"for current state information: "
                f">={self._classification_points[last_idx]}"
            )

        m = getattr(self.estimator, "n_jobs", None)
        threads = self._threads_to_use if m is None else 1

        # compute all new updates since then
        out = Parallel(n_jobs=threads)(
            delayed(self._predict_proba_for_estimator)(
                X,
                i,
            )
            for i in range(last_idx, next_idx)
        )
        probas, preds = zip(*out)

        # only compute new indices
        for i in range(last_idx, next_idx):
            accept_decision, state_info = self._decide_prediction_safety(
                i,
                probas[i],
                preds[i],
                state_info,
            )

        probas = np.array(
            [
                probas[max(0, state_info[i][0] - last_idx)][i]
                if accept_decision[i]
                else [-1 for _ in range(self.n_classes_)]
                for i in range(n_instances)
            ]
        )

        self.state_info = state_info

        return probas, accept_decision

    def _score(self, X, y) -> Tuple[float, float, float]:
        self._predict(X)

        accuracy = np.average(
            [
                self.state_info[i][2] == self._class_dictionary[y[i]]
                for i in range(len(self.state_info))
            ]
        )
        earliness = np.average(
            [
                self._classification_points[self.state_info[i][0]] / self.series_length_
                for i in range(len(self.state_info))
            ]
        )
        return (
            (2 * accuracy * (1 - earliness)) / (accuracy + (1 - earliness)),
            accuracy,
            earliness,
        )

    def _decide_prediction_safety(self, idx, probas, preds, state_info):
        # stores whether we have made a final decision on a prediction, if true
        # state info won't be edited in later time stamps
        finished = state_info[:, 1] >= self.consecutive_predictions
        n_instances = len(preds)

        full_length_ts = idx == len(self._classification_points) - 1
        if full_length_ts:
            accept_decision = np.ones(n_instances, dtype=bool)
        else:
            offsets = np.argwhere(finished == 0).flatten()
            accept_decision = np.ones(n_instances, dtype=bool)
            if len(offsets) > 0:
                decisions_subset = (
                    probas[offsets][preds[offsets]] >= self.probability_threshold
                )
                accept_decision[offsets] = decisions_subset

        # record consecutive class decisions
        state_info = np.array(
            [
                self._update_state_info(accept_decision, preds, state_info, i, idx)
                if not finished[i]
                else state_info[i]
                for i in range(n_instances)
            ]
        )

        # check safety of decisions
        if full_length_ts:
            # Force prediction at last time stamp
            accept_decision = np.ones(n_instances, dtype=bool)
        else:
            accept_decision = state_info[:, 1] >= self.consecutive_predictions

        return accept_decision, state_info

    def _fit_estimator(self, X, y, i):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (i + 1)
        rng = check_random_state(rs)

        estimator = _clone_estimator(
            DrCIF() if self.estimator is None else self.estimator,
            rng,
        )

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            estimator.n_jobs = self._threads_to_use

        estimator.fit(X[:, :, : self._classification_points[i]], y)

        return estimator

    def _predict_proba_for_estimator(self, X, i):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (i + 1)
        rng = check_random_state(rs)

        probas = self._estimators[i].predict_proba(
            X[:, :, : self._classification_points[i]]
        )
        preds = np.array(
            [int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in probas]
        )

        return probas, preds

    def _get_next_idx(self, series_length):
        """Return the largest index smaller than the series length."""
        next_idx = -1
        for idx, offset in enumerate(np.sort(self._classification_points)):
            if offset <= series_length:
                next_idx = idx
        return next_idx

    def _update_state_info(self, accept_decision, preds, state_info, idx, time_stamp):
        # consecutive predictions, add one if positive decision and same class
        if accept_decision[idx] and preds[idx] == state_info[idx][2]:
            return (
                time_stamp,
                state_info[idx][1] + 1,
                preds[idx],
                self._classification_points[time_stamp],
            )
        # set to 0 if the decision is negative, 1 if its positive but different class
        else:
            return (
                time_stamp,
                1 if accept_decision[idx] else 0,
                preds[idx],
                self._classification_points[time_stamp],
            )

    def _proba_output_to_preds(self, out):
        rng = check_random_state(self.random_state)
        preds = np.array(
            [
                self.classes_[
                    int(rng.choice(np.flatnonzero(out[0][i] == out[0][i].max())))
                ]
                if out[1][i]
                else -1
                for i in range(len(out[0]))
            ]
        )
        return preds, out[1]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        from sktime.classification.feature_based import SummaryClassifier
        from sktime.classification.interval_based import TimeSeriesForestClassifier

        if parameter_set == "results_comparison":
            return {
                "classification_points": [6, 10, 16, 24],
                "estimator": TimeSeriesForestClassifier(n_estimators=10),
            }
        else:
            return {
                "classification_points": [3, 5],
                "estimator": SummaryClassifier(
                    estimator=RandomForestClassifier(n_estimators=2)
                ),
            }
