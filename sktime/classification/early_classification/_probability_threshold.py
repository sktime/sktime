"""Probability Threshold Early Classifier.

An early classifier using a prediction probability threshold with a time series
classifier.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ProbabilityThresholdEarlyClassifier"]

import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.utils.validation.panel import check_X


# TODO: fix this in 0.25.0
# base class should have been changed to BaseEarlyClassifier
class ProbabilityThresholdEarlyClassifier(BaseClassifier):
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
        CanonicalIntervalForest.
    classification_points : List or None, default=None
        List of integer time series time stamps to build classifiers and allow
        predictions at. Early predictions must have a series length that matches a value
        in the _classification_points List. Duplicate values will be removed, and the
        full series length will be appended if not present.
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
        probability_threshold=0.85,
        consecutive_predictions=1,
        estimator=None,
        classification_points=None,
        n_jobs=1,
        random_state=None,
    ):
        self.probability_threshold = probability_threshold
        self.consecutive_predictions = consecutive_predictions
        self.estimator = estimator
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []

        super().__init__()

    def _fit(self, X, y):
        m = getattr(self.estimator, "predict_proba", None)
        if not callable(m):
            raise ValueError("Base estimator must have a predict_proba method.")

        _, _, series_length = X.shape

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

        self._estimators = Parallel(n_jobs=threads)(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
            )
            for i in range(len(self._classification_points))
        )

        return self

    def _predict(self, X) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        _, _, series_length = X.shape
        idx = self._classification_point_dictionary.get(series_length, -1)
        if idx == -1:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Current classification points: {self._classification_points}"
            )

        return self._estimators[idx].predict_proba(X)

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

        # make a decision based on probability threshold, record consecutive class
        # decisions
        decisions = [
            X_probabilities[i][preds[i]] >= self.probability_threshold
            for i in range(n_instances)
        ]
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
        if self.consecutive_predictions < 2:
            return decisions, new_state_info
        else:
            return [
                True if new_state_info[i][1] >= self.consecutive_predictions else False
                for i in range(n_instances)
            ], new_state_info

    def _fit_estimator(self, X, y, i):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = None if self.random_state is None else rs * 37 * (i + 1) % 2**31
        rng = check_random_state(rs)

        estimator = _clone_estimator(
            CanonicalIntervalForest() if self.estimator is None else self.estimator,
            rng,
        )

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            estimator.n_jobs = self._threads_to_use

        estimator.fit(X[:, :, : self._classification_points[i]], y)

        return estimator

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
        from sktime.classification.dummy import DummyClassifier
        from sktime.classification.feature_based import Catch22Classifier
        from sktime.utils.validation._dependencies import _check_soft_dependencies

        if _check_soft_dependencies("numba", severity="none"):
            est = Catch22Classifier(estimator=RandomForestClassifier(n_estimators=2))
        else:
            est = DummyClassifier()

        params1 = {"classification_points": [3], "estimator": est}
        params2 = {"probability_threshold": 0.9, "estimator": est}
        return [params1, params2]
