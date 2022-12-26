# -*- coding: utf-8 -*-
"""ContractableBOSS classifier.

Dictionary based cBOSS classifier based on SFA transform. Improves the
ensemble structure of the original BOSS algorithm.
"""

__author__ = ["MatthewMiddlehurst", "BINAYKUMAR943"]

__all__ = ["ContractableBOSS", "pairwise_distances"]

import math
import time
import warnings

import numpy as np
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import IndividualBOSS
from sktime.classification.dictionary_based._boss import pairwise_distances
from sktime.utils.validation.panel import check_X_y


class ContractableBOSS(BaseClassifier):
    """Contractable Bag of Symbolic Fourier Approximation Symbols (cBOSS).

    Implementation of BOSS Ensemble from Schäfer (2015) with refinements
    described in Middlehurst, Vickers and Bagnall (2019). [1, 2]_

    Overview: Input "n" series of length "m" and cBOSS randomly samples
    `n_parameter_samples` parameter sets, evaluting each with LOOCV. It then
    retains `max_ensemble_size` classifiers with the highest accuracy
    There are three primary parameters:
        - alpha: alphabet size
        - w: window length
        - l: word length.

    For any combination, a single BOSS slides a window length "w" along the
    series. The "w" length window is shortened to an "l" length word by
    taking a Fourier transform and keeping the first l/2 complex coefficients.
    These "l" coefficients are then discretised into "alpha" possible values,
    to form a word length "l". A histogram of words for each
    series is formed and stored.

    Fit involves finding "n" histograms.

    Predict uses 1 nearest neighbor with a bespoke BOSS distance function.

    Parameters
    ----------
    n_parameter_samples : int, default = 250
        If search is randomised, number of parameter combos to try.
    max_ensemble_size : int or None, default = 50
        Maximum number of classifiers to retain. Will limit number of retained
        classifiers even if more than `max_ensemble_size` are within threshold.
    max_win_len_prop : int or float, default = 1
        Maximum window length as a proportion of the series length.
    min_window : int, default = 10
        Minimum window size.
    time_limit_in_minutes : int, default = 0
        Time contract to limit build time in minutes. Default of 0 means no limit.
    contract_max_n_parameter_samples : int, default=np.inf
        Max number of parameter combinations to consider when time_limit_in_minutes is
        set.
    typed_dict : bool, default="deprecated"
        Use a numba TypedDict to store word counts. May increase memory usage, but will
        be faster for larger datasets. As the Dict cannot be pickled currently, there
        will be some overhead converting it to a python dict with multiple threads and
        pickling.

        .. deprecated:: 0.13.3
            ``typed_dict`` was deprecated in version 0.13.3 and will be removed in 0.15.

    save_train_predictions : bool, default=False
        Save the ensemble member train predictions in fit for use in _get_train_probs
        leave-one-out cross-validation.
    n_jobs : int, default = 1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    feature_selection: {"chi2", "none", "random"}, default: none
        Sets the feature selections strategy to be used. Chi2 reduces the number
        of words significantly and is thus much faster (preferred). Random also reduces
        the number significantly. None applies not feature selectiona and yields large
        bag of words, e.g. much memory may be needed.
    random_state : int or None, default=None
        Seed for random integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : list
        The classes labels.
    n_instances_ : int
        Number of instances. Extracted from the data.
    n_estimators_ : int
        The final number of classifiers used. Will be <= `max_ensemble_size` if
        `max_ensemble_size` has been specified.
    series_length_ : int
        Length of all series (assumed equal).
    estimators_ : list
       List of DecisionTree classifiers.
    weights_ :
        Weight of each classifier in the ensemble.

    See Also
    --------
    BOSSEnsemble, IndividualBOSS

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/cBOSS.java>`_.

    References
    ----------
    .. [1] Patrick Schäfer, "The BOSS is concerned with time series classification
       in the presence of noise", Data Mining and Knowledge Discovery, 29(6): 2015
       https://link.springer.com/article/10.1007/s10618-014-0377-7

    .. [2] Matthew Middlehurst, William Vickers and Anthony Bagnall
       "Scalable Dictionary Classifiers for Time Series Classification",
       in proc 20th International Conference on Intelligent Data Engineering
       and Automated Learning,LNCS, volume 11871
       https://link.springer.com/chapter/10.1007/978-3-030-33607-3_2

    Examples
    --------
    >>> from sktime.classification.dictionary_based import ContractableBOSS
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = ContractableBOSS(
    ...     n_parameter_samples=10, max_ensemble_size=3
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    ContractableBOSS(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "classifier_type": "dictionary",
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        n_parameter_samples=250,
        max_ensemble_size=50,
        max_win_len_prop=1,
        min_window=10,
        time_limit_in_minutes=0.0,
        contract_max_n_parameter_samples=np.inf,
        typed_dict="deprecated",
        save_train_predictions=False,
        feature_selection="none",
        n_jobs=1,
        random_state=None,
    ):
        self.n_parameter_samples = n_parameter_samples
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.min_window = min_window

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_parameter_samples = contract_max_n_parameter_samples
        self.typed_dict = typed_dict
        self.save_train_predictions = save_train_predictions
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.feature_selection = feature_selection

        self.estimators_ = []
        self.weights_ = []
        self.n_estimators_ = 0
        self.series_length_ = 0
        self.n_instances_ = 0

        self._weight_sum = 0
        self._word_lengths = [16, 14, 12, 10, 8]
        self._norm_options = [True, False]
        self._alphabet_size = 4

        super(ContractableBOSS, self).__init__()

    def _fit(self, X, y):
        """Fit a cBOSS ensemble on cases (X,y), where y is the target variable.

        Build an ensemble of BOSS classifiers from the training set (X,
        y), through randomising over the para space to make a fixed size
        ensemble of the best.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
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
        ending in "_" and sets is_fitted flag to True.
        """
        time_limit = self.time_limit_in_minutes * 60
        self.n_instances_, _, self.series_length_ = X.shape

        self.estimators_ = []
        self.weights_ = []

        if self.typed_dict != "deprecated":
            warnings.warn(
                "``typed_dict`` was deprecated in version 0.13.3 and "
                "will be removed in 0.15."
            )

        # Window length parameter space dependent on series length
        max_window_searches = self.series_length_ / 4
        max_window = int(self.series_length_ * self.max_win_len_prop)
        win_inc = int((max_window - self.min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1
        if self.min_window > max_window + 1:
            raise ValueError(
                f"Error in ContractableBOSS, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={max_window}."
                f" Try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )
        possible_parameters = self._unique_parameters(max_window, win_inc)
        num_classifiers = 0
        start_time = time.time()
        train_time = 0
        subsample_size = int(self.n_instances_ * 0.7)
        lowest_acc = 1
        lowest_acc_idx = 0

        rng = check_random_state(self.random_state)

        if time_limit > 0:
            n_parameter_samples = 0
            contract_max_n_parameter_samples = self.contract_max_n_parameter_samples
        else:
            n_parameter_samples = self.n_parameter_samples
            contract_max_n_parameter_samples = np.inf

        while (
            (
                train_time < time_limit
                and num_classifiers < contract_max_n_parameter_samples
            )
            or num_classifiers < n_parameter_samples
        ) and len(possible_parameters) > 0:
            parameters = possible_parameters.pop(
                rng.randint(0, len(possible_parameters))
            )

            subsample = rng.choice(
                self.n_instances_, size=subsample_size, replace=False
            )
            X_subsample = X[subsample]
            y_subsample = y[subsample]

            boss = IndividualBOSS(
                *parameters,
                alphabet_size=self._alphabet_size,
                save_words=False,
                n_jobs=self.n_jobs,
                feature_selection=self.feature_selection,
                random_state=self.random_state,
            )
            boss.fit(X_subsample, y_subsample)
            boss._clean()
            boss._subsample = subsample

            boss._accuracy = self._individual_train_acc(
                boss,
                y_subsample,
                subsample_size,
                0 if num_classifiers < self.max_ensemble_size else lowest_acc,
            )
            if boss._accuracy > 0:
                weight = math.pow(boss._accuracy, 4)
            else:
                weight = 0.000000001

            # Only keep the classifier, if its accuracy is non-zero
            if boss._accuracy > 0:
                if num_classifiers < self.max_ensemble_size:
                    if boss._accuracy < lowest_acc:
                        lowest_acc = boss._accuracy
                        lowest_acc_idx = num_classifiers
                    self.weights_.append(weight)
                    self.estimators_.append(boss)
                elif boss._accuracy > lowest_acc:
                    self.weights_[lowest_acc_idx] = weight
                    self.estimators_[lowest_acc_idx] = boss
                    lowest_acc, lowest_acc_idx = self._worst_ensemble_acc()

                num_classifiers += 1
            train_time = time.time() - start_time

        self.n_estimators_ = len(self.estimators_)
        self._weight_sum = np.sum(self.weights_)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        sums = np.zeros((X.shape[0], self.n_classes_))

        for n, clf in enumerate(self.estimators_):
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self._class_dictionary[preds[i]]] += self.weights_[n]

        dists = sums / (np.ones(self.n_classes_) * self._weight_sum)

        return dists

    def _worst_ensemble_acc(self):
        min_acc = 1.0
        min_acc_idx = -1

        for c, classifier in enumerate(self.estimators_):
            if classifier._accuracy < min_acc:
                min_acc = classifier._accuracy
                min_acc_idx = c

        return min_acc, min_acc_idx

    def _unique_parameters(self, max_window, win_inc):
        possible_parameters = [
            [win_size, word_len, normalise]
            for n, normalise in enumerate(self._norm_options)
            for win_size in range(self.min_window, max_window + 1, win_inc)
            for g, word_len in enumerate(self._word_lengths)
        ]

        return possible_parameters

    def _get_train_probs(self, X, y) -> np.ndarray:
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_numpy=True, enforce_univariate=True)

        n_instances, _, series_length = X.shape

        if n_instances != self.n_instances_ or series_length != self.series_length_:
            raise ValueError(
                "n_instances, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        results = np.zeros((n_instances, self.n_classes_))
        divisors = np.zeros(n_instances)

        if self.save_train_predictions:
            for i, clf in enumerate(self.estimators_):
                subsample = clf._subsample
                preds = clf._train_predictions

                for n, pred in enumerate(preds):
                    results[subsample[n]][
                        self._class_dictionary[pred]
                    ] += self.weights_[i]
                    divisors[subsample[n]] += self.weights_[i]

        else:
            for i, clf in enumerate(self.estimators_):
                subsample = clf._subsample
                distance_matrix = pairwise_distances(
                    clf._transformed_data, n_jobs=self.n_jobs
                )

                preds = []
                for j in range(len(subsample)):
                    preds.append(clf._train_predict(j, distance_matrix))

                for n, pred in enumerate(preds):
                    results[subsample[n]][
                        self._class_dictionary[pred]
                    ] += self.weights_[i]
                    divisors[subsample[n]] += self.weights_[i]

        for i in range(n_instances):
            results[i] = (
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _individual_train_acc(self, boss, y, train_size, lowest_acc):
        correct = 0
        required_correct = int(lowest_acc * train_size)

        # there may be no words if feature selection is too aggressive
        if boss._transformed_data.shape[1] > 0:
            distance_matrix = pairwise_distances(
                boss._transformed_data, n_jobs=self.n_jobs
            )

            for i in range(train_size):
                if correct + train_size - i < required_correct:
                    return -1

                c = boss._train_predict(i, distance_matrix)
                if c == y[i]:
                    correct += 1

                if self.save_train_predictions:
                    boss._train_predictions.append(c)

        return correct / train_size

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"n_parameter_samples": 10, "max_ensemble_size": 5}
        else:
            return {
                "n_parameter_samples": 4,
                "max_ensemble_size": 2,
                "save_train_predictions": True,
                "feature_selection": "none",
            }
