# -*- coding: utf-8 -*-
"""BOSS classifiers.

Dictionary based BOSS classifiers based on SFA transform. Contains a single
BOSS and a BOSS ensemble.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["BOSSEnsemble", "IndividualBOSS", "boss_distance"]

import sys
from itertools import compress

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA
from sktime.utils.validation.panel import check_X_y


class BOSSEnsemble(BaseClassifier):
    """Ensemble of bag of Symbolic Fourier Approximation Symbols (BOSS).

    Implementation of BOSS Ensemble from Schäfer (2015). [1]_

    Overview: Input "n" series of length "m" and BOSS performs a grid search over
    a set of parameter values, evaluating each with a LOOCV. It then retains
    all ensemble members within 92% of the best by default for use in the ensemble.
    There are three primary parameters:
        - alpha: alphabet size
        - w: window length
        - l: word length.

    For any combination, a single BOSS slides a window length "w" along the
    series. The w length window is shortened to an "l" length word through
    taking a Fourier transform and keeping the first l/2 complex coefficients.
    These "l" coefficients are then discretized into alpha possible values,
    to form a word length "l". A histogram of words for each
    series is formed and stored.

    Fit involves finding "n" histograms.

    Predict uses 1 nearest neighbor with a bespoke BOSS distance function.

    Parameters
    ----------
    threshold : float, default=0.92
        Threshold used to determine which classifiers to retain. All classifiers
        within percentage `threshold` of the best one are retained.
    max_ensemble_size : int or None, default=500
        Maximum number of classifiers to retain. Will limit number of retained
        classifiers even if more than `max_ensemble_size` are within threshold.
    max_win_len_prop : int or float, default=1
        Maximum window length as a proportion of the series length.
    min_window : int, default=10
        Minimum window size.
    save_train_predictions : bool, default=False
        Save the ensemble member train predictions in fit for use in _get_train_probs
        leave-one-out cross-validation.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

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

    See Also
    --------
    IndividualBOSS, ContractableBOSS

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/BOSS.java>`_.

    References
    ----------
    .. [1] Patrick Schäfer, "The BOSS is concerned with time series classification
       in the presence of noise", Data Mining and Knowledge Discovery, 29(6): 2015
       https://link.springer.com/article/10.1007/s10618-014-0377-7

    Examples
    --------
    >>> from sktime.classification.dictionary_based import BOSSEnsemble
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = BOSSEnsemble(max_ensemble_size=5)
    >>> clf.fit(X_train, y_train)
    BOSSEnsemble(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:train_estimate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        threshold=0.92,
        max_ensemble_size=500,
        max_win_len_prop=1,
        min_window=10,
        save_train_predictions=False,
        n_jobs=1,
        random_state=None,
    ):
        self.threshold = threshold
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.min_window = min_window

        self.save_train_predictions = save_train_predictions
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_ = []
        self.n_estimators_ = 0
        self.series_length_ = 0
        self.n_instances_ = 0

        self._word_lengths = [16, 14, 12, 10, 8]
        self._norm_options = [True, False]
        self._alphabet_size = 4

        super(BOSSEnsemble, self).__init__()

    def _fit(self, X, y):
        """Fit a boss ensemble on cases (X,y), where y is the target variable.

        Build an ensemble of BOSS classifiers from the training set (X,
        y), through  creating a variable size ensemble of those within a
        threshold of the best.

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
        self.n_instances_, _, self.series_length_ = X.shape

        self.estimators_ = []

        # Window length parameter space dependent on series length
        max_window_searches = self.series_length_ / 4

        max_window = int(self.series_length_ * self.max_win_len_prop)
        win_inc = int((max_window - self.min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1
        if self.min_window > max_window + 1:
            raise ValueError(
                f"Error in BOSSEnsemble, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={max_window}."
                f" Try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )
        max_acc = -1
        min_max_acc = -1
        for normalise in self._norm_options:
            for win_size in range(self.min_window, max_window + 1, win_inc):
                boss = IndividualBOSS(
                    win_size,
                    self._word_lengths[0],
                    normalise,
                    self._alphabet_size,
                    save_words=True,
                    n_jobs=self._threads_to_use,
                    random_state=self.random_state,
                )
                boss.fit(X, y)

                best_classifier_for_win_size = boss
                best_acc_for_win_size = -1

                # the used word length may be shorter
                best_word_len = boss._transformer.word_length

                for n, word_len in enumerate(self._word_lengths):
                    if n > 0:
                        boss = boss._shorten_bags(word_len)

                    boss._accuracy = self._individual_train_acc(
                        boss, y, self.n_instances_, best_acc_for_win_size
                    )

                    if boss._accuracy >= best_acc_for_win_size:
                        best_acc_for_win_size = boss._accuracy
                        best_classifier_for_win_size = boss
                        best_word_len = word_len

                if self._include_in_ensemble(
                    best_acc_for_win_size,
                    max_acc,
                    min_max_acc,
                    len(self.estimators_),
                ):
                    best_classifier_for_win_size._clean()
                    best_classifier_for_win_size._set_word_len(best_word_len)
                    self.estimators_.append(best_classifier_for_win_size)

                    if best_acc_for_win_size > max_acc:
                        max_acc = best_acc_for_win_size
                        self.estimators_ = list(
                            compress(
                                self.estimators_,
                                [
                                    classifier._accuracy >= max_acc * self.threshold
                                    for c, classifier in enumerate(self.estimators_)
                                ],
                            )
                        )

                    min_max_acc, min_acc_ind = self._worst_ensemble_acc()

                    if len(self.estimators_) > self.max_ensemble_size:
                        if min_acc_ind > -1:
                            del self.estimators_[min_acc_ind]
                            min_max_acc, min_acc_ind = self._worst_ensemble_acc()

        self.n_estimators_ = len(self.estimators_)

        return self

    def _predict(self, X):
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

    def _predict_proba(self, X):
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

        for clf in self.estimators_:
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self._class_dictionary[preds[i]]] += 1
        dists = sums / (np.ones(self.n_classes_) * self.n_estimators_)

        return dists

    def _include_in_ensemble(self, acc, max_acc, min_max_acc, size):
        if acc >= max_acc * self.threshold:
            if size >= self.max_ensemble_size:
                return acc > min_max_acc
            else:
                return True
        return False

    def _worst_ensemble_acc(self):
        min_acc = 1.0
        min_acc_idx = -1

        for c, classifier in enumerate(self.estimators_):
            if classifier._accuracy < min_acc:
                min_acc = classifier._accuracy
                min_acc_idx = c

        return min_acc, min_acc_idx

    def _get_train_probs(self, X, y):
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

        for i, clf in enumerate(self.estimators_):
            preds = (
                clf._train_predictions
                if self.save_train_predictions
                else Parallel(n_jobs=self._threads_to_use)(
                    delayed(clf._train_predict)(
                        i,
                    )
                    for i in range(n_instances)
                )
            )

            for n, pred in enumerate(preds):
                results[n][self._class_dictionary[pred]] += 1
                divisors[n] += 1

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

        if self._threads_to_use > 1:
            c = Parallel(n_jobs=self._threads_to_use)(
                delayed(boss._train_predict)(
                    i,
                )
                for i in range(train_size)
            )

            for i in range(train_size):
                if correct + train_size - i < required_correct:
                    return -1
                elif c[i] == y[i]:
                    correct += 1

                if self.save_train_predictions:
                    boss._train_predictions.append(c[i])
        else:
            for i in range(train_size):
                if correct + train_size - i < required_correct:
                    return -1

                c = boss._train_predict(i)

                if c == y[i]:
                    correct += 1

                if self.save_train_predictions:
                    boss._train_predictions.append(c)

        return correct / train_size


class IndividualBOSS(BaseClassifier):
    """Single bag of Symbolic Fourier Approximation Symbols (IndividualBOSS).

    Bag of SFA Symbols Ensemble: implementation of a single BOSS Schaffer, the base
    classifier for the boss ensemble.

    Implementation of single BOSS model from Schäfer (2015). [1]_

    This is the underlying classifier for each classifier in the BOSS ensemble.

    Overview: input "n" series of length "m" and IndividualBoss performs a SFA
    transform to form a sparse dictionary of discretised words. The resulting
    dictionary is used with the BOSS distance function in a 1-nearest neighbor.

    Fit involves finding "n" histograms.

    Predict uses 1 nearest neighbor with a bespoke BOSS distance function.

    Parameters
    ----------
    window_size : int
        Size of the window to use in BOSS algorithm.
    word_length : int
        Length of word to use to use in BOSS algorithm.
    norm : bool, default = False
        Whether to normalize words by dropping the first Fourier coefficient.
    alphabet_size : default = 4
        Number of possible letters (values) for each word.
    save_words : bool, default = True
        Whether to keep NumPy array of words in SFA transformation even after
        the dictionary of words is returned. If True, the array is saved, which
        can shorten the time to calculate dictionaries using a shorter
        `word_length` (since the last "n" letters can be removed).
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : list
        The classes labels.

    See Also
    --------
    BOSSEnsemble, ContractableBOSS

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/IndividualBOSS.java>`_.

    References
    ----------
    .. [1] Patrick Schäfer, "The BOSS is concerned with time series classification
       in the presence of noise", Data Mining and Knowledge Discovery, 29(6): 2015
       https://link.springer.com/article/10.1007/s10618-014-0377-7

    Examples
    --------
    >>> from sktime.classification.dictionary_based import IndividualBOSS
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = IndividualBOSS()
    >>> clf.fit(X_train, y_train)
    IndividualBOSS(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
    }

    def __init__(
        self,
        window_size=10,
        word_length=8,
        norm=False,
        alphabet_size=4,
        save_words=False,
        n_jobs=1,
        random_state=None,
    ):
        self.window_size = window_size
        self.word_length = word_length
        self.norm = norm
        self.alphabet_size = alphabet_size

        self.save_words = save_words
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._transformer = None
        self._transformed_data = []
        self._class_vals = []
        self._accuracy = 0
        self._subsample = []
        self._train_predictions = []

        super(IndividualBOSS, self).__init__()

    def _fit(self, X, y):
        """Fit a single boss classifier on n_instances cases (X,y).

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
        self._transformer = SFA(
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            norm=self.norm,
            remove_repeat_words=True,
            bigrams=False,
            save_words=self.save_words,
            n_jobs=self._threads_to_use,
        )

        sfa = self._transformer.fit_transform(X)
        self._transformed_data = sfa[0]
        self._class_vals = y

        return self

    def _predict(self, X):
        """Predict class values of all instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        test_bags = self._transformer.transform(X)
        test_bags = test_bags[0]

        classes = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._test_nn)(
                test_bag,
            )
            for test_bag in test_bags
        )

        return np.array(classes)

    def _test_nn(self, test_bag):
        rng = check_random_state(self.random_state)

        best_dist = sys.float_info.max
        nn = None

        for n, bag in enumerate(self._transformed_data):
            dist = boss_distance(test_bag, bag, best_dist)

            if dist < best_dist or (dist == best_dist and rng.random() < 0.5):
                best_dist = dist
                nn = self._class_vals[n]

        return nn

    def _train_predict(self, train_num):
        test_bag = self._transformed_data[train_num]
        best_dist = sys.float_info.max
        nn = None

        for n, bag in enumerate(self._transformed_data):
            if n == train_num:
                continue

            dist = boss_distance(test_bag, bag, best_dist)

            if dist < best_dist:
                best_dist = dist
                nn = self._class_vals[n]

        return nn

    def _shorten_bags(self, word_len):
        new_boss = IndividualBOSS(
            self.window_size,
            word_len,
            self.norm,
            self.alphabet_size,
            save_words=self.save_words,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        new_boss._transformer = self._transformer
        sfa = self._transformer._shorten_bags(word_len)
        new_boss._transformed_data = sfa[0]

        new_boss._class_vals = self._class_vals
        new_boss.n_classes_ = self.n_classes_
        new_boss.classes_ = self.classes_
        new_boss._class_dictionary = self._class_dictionary

        new_boss._threads_to_use = self._threads_to_use
        new_boss._is_fitted = True
        return new_boss

    def _clean(self):
        self._transformer.words = None
        self._transformer.save_words = False

    def _set_word_len(self, word_len):
        self.word_length = word_len
        self._transformer.word_length = word_len


def boss_distance(first, second, best_dist=sys.float_info.max):
    """Find the distance between two histograms.

    This returns the distance between first and second dictionaries, using a non
    symmetric distance measure. It is used to find the distance between historgrams
    of words.

    This distance function is designed for sparse matrix, represented as either a
    dictionary or an arrray. It only measures the distance between counts present in
    the first dictionary and the second. Hence dist(a,b) does not necessarily equal
    dist(b,a).

    Parameters
    ----------
    first : dict
        Base dictionary used in distance measurement.
    second : dict
        Second dictionary that will be used to measure distance from `first`.
    best_dist : int, float or sys.float_info.max
        Largest distance value. Values above this will be replaced by
        sys.float_info.max.

    Returns
    -------
    dist : float
        The boss distance between the first and second dictionaries.
    """
    dist = 0

    if isinstance(first, dict):
        for word, val_a in first.items():
            val_b = second.get(word, 0)
            buf = val_a - val_b
            dist += buf * buf

            if dist > best_dist:
                return sys.float_info.max
    else:
        dist = np.sum(
            [
                0 if first[i] == 0 else (first[i] - second[i]) * (first[i] - second[i])
                for i in range(len(first))
            ]
        )

    return dist
