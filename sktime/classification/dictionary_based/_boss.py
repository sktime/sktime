# -*- coding: utf-8 -*-
"""BOSS classifiers.

dictionary based BOSS classifiers based on SFA transform. Contains a single
BOSS and a BOSS ensemble.
"""

__author__ = "Matthew Middlehurst"
__all__ = ["BOSSEnsemble", "IndividualBOSS", "boss_distance"]

import sys
from itertools import compress

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA
from sktime.utils.validation.panel import check_X, check_X_y


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
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes : int
        Number of classes. Extracted from the data.
    n_instances : int
        Number of instances. Extracted from the data.
    n_estimators : int
        The final number of classifiers used. Will be <= `max_ensemble_size` if
        `max_ensemble_size` has been specified.
    series_length : int
        Length of all series (assumed equal).
    classifiers : list
       List of DecisionTree classifiers.
    class_dictionary: dict
        Dictionary of classes. Extracted from the data.


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
    >>> from sktime.datasets import load_italy_power_demand
    >>> X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    >>> X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    >>> clf = BOSSEnsemble()
    >>> clf.fit(X_train, y_train)
    BOSSEnsemble(...)
    >>> y_pred = clf.predict(X_test)
    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": True,
        "contractable": False,
    }

    def __init__(
        self,
        threshold=0.92,
        max_ensemble_size=500,
        max_win_len_prop=1,
        min_window=10,
        n_jobs=1,
        random_state=None,
    ):
        self.threshold = threshold
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifiers = []
        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        self.n_estimators = 0
        self.series_length = 0
        self.n_instances = 0

        self.word_lengths = [16, 14, 12, 10, 8]
        self.norm_options = [True, False]
        self.min_window = min_window
        self.alphabet_size = 4
        super(BOSSEnsemble, self).__init__()

    def fit(self, X, y):
        """Fit a boss ensemble on cases (X,y), where y is the target variable.

        Build an ensemble of BOSS classifiers from the training set (X,
        y), through  creating a variable size ensemble of those within a
        threshold of the best.

        Parameters
        ----------
        X : pd.DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)

        self.n_instances, _, self.series_length = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifiers = []

        # Window length parameter space dependent on series length
        max_window_searches = self.series_length / 4

        max_window = int(self.series_length * self.max_win_len_prop)
        win_inc = int((max_window - self.min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1
        if self.min_window > max_window + 1:
            raise ValueError(
                f"Error in BOSSEnsemble, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={max_window},"
                f" series length is {self.series_length}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )
        max_acc = -1
        min_max_acc = -1
        for normalise in self.norm_options:
            for win_size in range(self.min_window, max_window + 1, win_inc):
                boss = IndividualBOSS(
                    win_size,
                    self.word_lengths[0],
                    normalise,
                    self.alphabet_size,
                    save_words=True,
                    random_state=self.random_state,
                )
                boss.fit(X, y)

                best_classifier_for_win_size = boss
                best_acc_for_win_size = -1

                # the used word length may be shorter
                best_word_len = boss.transformer.word_length

                for n, word_len in enumerate(self.word_lengths):
                    if n > 0:
                        boss = boss._shorten_bags(word_len)

                    boss.accuracy = self._individual_train_acc(
                        boss, y, self.n_instances, best_acc_for_win_size
                    )

                    if boss.accuracy >= best_acc_for_win_size:
                        best_acc_for_win_size = boss.accuracy
                        best_classifier_for_win_size = boss
                        best_word_len = word_len

                if self._include_in_ensemble(
                    best_acc_for_win_size,
                    max_acc,
                    min_max_acc,
                    len(self.classifiers),
                ):
                    best_classifier_for_win_size._clean()
                    best_classifier_for_win_size._set_word_len(best_word_len)
                    self.classifiers.append(best_classifier_for_win_size)

                    if best_acc_for_win_size > max_acc:
                        max_acc = best_acc_for_win_size
                        self.classifiers = list(
                            compress(
                                self.classifiers,
                                [
                                    classifier.accuracy >= max_acc * self.threshold
                                    for c, classifier in enumerate(self.classifiers)
                                ],
                            )
                        )

                    min_max_acc, min_acc_ind = self._worst_ensemble_acc()

                    if len(self.classifiers) > self.max_ensemble_size:
                        if min_acc_ind > -1:
                            del self.classifiers[min_acc_ind]
                            min_max_acc, min_acc_ind = self._worst_ensemble_acc()

        self.n_estimators = len(self.classifiers)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, 1)

        Returns
        -------
        preds : np.ndarray of shape (n, 1)
            Predicted class.
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, 1)

        Returns
        -------
        dists : array of shape (n_instances, n_classes)
            Predicted probability of each class.
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

        sums = np.zeros((X.shape[0], self.n_classes))

        for clf in self.classifiers:
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self.class_dictionary[preds[i]]] += 1
        dists = sums / (np.ones(self.n_classes) * self.n_estimators)

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

        for c, classifier in enumerate(self.classifiers):
            if classifier.accuracy < min_acc:
                min_acc = classifier.accuracy
                min_acc_idx = c

        return min_acc, min_acc_idx

    def _get_train_probs(self, X):
        num_inst = X.shape[0]
        results = np.zeros((num_inst, self.n_classes))
        divisor = np.ones(self.n_classes) * self.n_estimators
        for i in range(num_inst):
            sums = np.zeros(self.n_classes)

            preds = Parallel(n_jobs=self.n_jobs)(
                delayed(clf._train_predict)(
                    i,
                )
                for clf in self.classifiers
            )

            for c in preds:
                sums[self.class_dictionary.get(c, -1)] += 1

            dists = sums / divisor
            for n in range(self.n_classes):
                results[i][n] = dists[n]

        return results

    def _individual_train_acc(self, boss, y, train_size, lowest_acc):
        correct = 0
        required_correct = int(lowest_acc * train_size)

        if self.n_jobs > 1:
            c = Parallel(n_jobs=self.n_jobs)(
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
        else:
            for i in range(train_size):
                if correct + train_size - i < required_correct:
                    return -1

                c = boss._train_predict(i)

                if c == y[i]:
                    correct += 1

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
    n_classes : int
        Number of classes. Extracted from the data.
    n_instances : int
        Number of instances. Extracted from the data.
    n_estimators : int
        The final number of classifiers used. Will be <= `max_ensemble_size` if
        `max_ensemble_size` has been specified.
    series_length : int
        Length of all series (assumed equal).
    class_dictionary: dict
        Dictionary of classes. Extracted from the data.

    See Also
    --------
    BOSSEnsemble, ContractableBOSS

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
    >>> from sktime.classification.dictionary_based import IndividualBOSS
    >>> from sktime.datasets import load_italy_power_demand
    >>> X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    >>> X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    >>> clf = IndividualBOSS()
    >>> clf.fit(X_train, y_train)
    IndividualBOSS(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        window_size=10,
        word_length=8,
        norm=False,
        alphabet_size=4,
        save_words=True,
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

        self.transformer = SFA(
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=window_size,
            norm=norm,
            remove_repeat_words=True,
            bigrams=False,
            save_words=save_words,
            n_jobs=n_jobs,
        )
        self.transformed_data = []
        self.accuracy = 0
        self.subsample = []

        self.class_vals = []
        self.num_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        super(IndividualBOSS, self).__init__()

    def fit(self, X, y):
        """Fit a single boss classifier on n_instances cases (X,y).

        Parameters
        ----------
        X : pd.DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)

        sfa = self.transformer.fit_transform(X)
        self.transformed_data = sfa[0]

        self.class_vals = y
        self.num_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict class values of all instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, 1]
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

        test_bags = self.transformer.transform(X)
        test_bags = test_bags[0]

        classes = Parallel(n_jobs=self.n_jobs)(
            delayed(self._test_nn)(
                test_bag,
            )
            for test_bag in test_bags
        )

        return np.array(classes)

    def predict_proba(self, X):
        """Predict class probabilities for all instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        dists : array of shape [n, self.n_classes]
        """
        preds = self.predict(X)
        dists = np.zeros((X.shape[0], self.num_classes))

        for i in range(0, X.shape[0]):
            dists[i, self.class_dictionary.get(preds[i])] += 1

        return dists

    def _test_nn(self, test_bag):
        rng = check_random_state(self.random_state)

        best_dist = sys.float_info.max
        nn = None

        for n, bag in enumerate(self.transformed_data):
            dist = boss_distance(test_bag, bag, best_dist)

            if dist < best_dist or (dist == best_dist and rng.random() < 0.5):
                best_dist = dist
                nn = self.class_vals[n]

        return nn

    def _train_predict(self, train_num):
        test_bag = self.transformed_data[train_num]
        best_dist = sys.float_info.max
        nn = None

        for n, bag in enumerate(self.transformed_data):
            if n == train_num:
                continue

            dist = boss_distance(test_bag, bag, best_dist)

            if dist < best_dist:
                best_dist = dist
                nn = self.class_vals[n]

        return nn

    def _shorten_bags(self, word_len):
        new_boss = IndividualBOSS(
            self.window_size,
            word_len,
            self.norm,
            self.alphabet_size,
            save_words=self.save_words,
            random_state=self.random_state,
        )
        new_boss.transformer = self.transformer
        sfa = self.transformer._shorten_bags(word_len)
        new_boss.transformed_data = sfa[0]

        new_boss.class_vals = self.class_vals
        new_boss.num_classes = self.num_classes
        new_boss.classes_ = self.classes_
        new_boss.class_dictionary = self.class_dictionary

        new_boss._is_fitted = True
        return new_boss

    def _clean(self):
        self.transformer.words = None
        self.transformer.save_words = False

    def _set_word_len(self, word_len):
        self.word_length = word_len
        self.transformer.word_length = word_len


# @njit()
# def _dist(val_a, val_b):
#     return (val_a - val_b) * (val_a - val_b)


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
