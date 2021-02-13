# -*- coding: utf-8 -*-
""" BOSS classifiers
dictionary based BOSS classifiers based on SFA transform. Contains a single
BOSS and a BOSS ensemble
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


# from numba import njit
# from numba.typed import Dict


class BOSSEnsemble(BaseClassifier):
    """Bag of SFA Symbols (BOSS)

    Bag of SFA Symbols Ensemble: implementation of BOSS from [1]

    Overview: Input n series length m
    BOSS performs a gird search over a set of parameter values, evaluating
    each with a LOOCV. It then retains
    all ensemble members within 92% of the best. There are three primary
    parameters:
            alpha: alphabet size
            w: window length
            l: word length.
    for any combination, a single BOSS slides a window length w along the
    series. The w length window is shortened to
    an l length word through taking a Fourier transform and keeping the
    first l/2 complex coefficients. These l
    coefficients are then discretised into alpha possible values, to form a
    word length l. A histogram of words for each
    series is formed and stored. fit involves finding n histograms.

    predict uses 1 nearest neighbour with a bespoke distance function.


    Parameters
    ----------
    threshold               : double [0,1]. retain all classifiers within
    threshold% of the best one, optional (default = 0.92)
    max_ensemble_size       : int or None, retain a maximum number of
    classifiers, even if within threshold, optional (default = 500)
    max_win_len_prop        : maximum window length as a proportion of
    series length (default = 1)
    min_window              : minimum window size, (default = 10)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    n_classes               : extracted from the data
    n_instances             : extracted from the data
    n_estimators            : The final number of classifiers used (
    <= max_ensemble_size)
    series_length           : length of all series (assumed equal)
    classifiers             : array of DecisionTree classifiers

    Notes
    -----
    ..[1] Patrick Schäfer, "The BOSS is concerned with time series classification
            in the presence of noise", Data Mining and Knowledge Discovery, 29(6): 2015
            https://link.springer.com/article/10.1007/s10618-014-0377-7

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/dictionary_based/BOSS.java
    """

    # Capabilities: data types this classifier can handle
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
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
        """Build an ensemble of BOSS classifiers from the training set (X,
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

                    # print("appending", best_acc_for_win_size, win_size)
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
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
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
    """Single Bag of SFA Symbols (BOSS) classifier

    Bag of SFA Symbols Ensemble: implementation of BOSS from Schaffer :
    @article
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
                0 if first[n] == 0 else (first[n] - second[n]) * (first[n] - second[n])
                for n in range(len(first))
            ]
        )

    return dist
