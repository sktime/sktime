""" BOSS classifiers
dictionary based BOSS classifiers based on SFA transform. Contains a single
BOSS and a BOSS ensemble
"""

__author__ = "Matthew Middlehurst"
__all__ = ["BOSSEnsemble", "BOSSIndividual", "boss_distance"]

import math
import sys
import time
from itertools import compress

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import class_distribution
from sklearn.utils import check_random_state
from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y

# from numba import njit
# from numba.typed import Dict

# TODO: Make more efficient


class BOSSEnsemble(BaseClassifier):
    """ Bag of SFA Symbols (BOSS)

    Bag of SFA Symbols Ensemble: implementation of BOSS from Schafer:
    @article
    {schafer15boss,
     author = {Patrick Schäfer,
            title = {The BOSS is concerned with time series classification
            in the presence of noise},
            journal = {Data Mining and Knowledge Discovery},
            volume = {29},
            number= {6},
            year = {2015}
    }
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
    coefficents are then discretised into alpha possible values, to form a
    word length l. A histogram of words for each
    series is formed and stored. fit involves finding n histograms.

    predict uses 1 nearest neighbour with a bespoke distance function.

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/dictionary_based/BOSS.java


    Parameters
    ----------
    randomised_ensemble     : bool, turns the option to just randomise the
    ensemble members rather than cross validate (cBOSS) (default=False)
    n_parameter_samples     : int, if search is randomised, number of
    parameter combos to try (default=250)
    threshold               : double [0,1]. retain all classifiers within
    threshold% of the best one, optional (default =0.92)
    max_ensemble_size       : int or None, retain a maximum number of
    classifiers, even if within threshold, optional
    (default = 500, recommended 50 for cBOSS)
    max_win_len_prop        : maximum window length as a proportion of
    series length (default =1)
    time_limit              : time contract to limit build time in minutes
    (default=0, no limit)
    min_window              : minimum window size, (default=10)
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    n_classes               : extracted from the data
    n_instances             : extracted from the data
    n_estimators           : The final number of classifiers used (
    <=max_ensemble_size)
    series_length           : length of all series (assumed equal)
    classifiers             : array of DecisionTree classifiers
    weights                 : weight of each classifier in the ensemble

    """

    def __init__(self,
                 randomised_ensemble=False,
                 n_parameter_samples=250,
                 threshold=0.92,
                 max_ensemble_size=500,
                 max_win_len_prop=1,
                 time_limit=0.0,
                 min_window=10,
                 random_state=None
                 ):
        self.randomised_ensemble = randomised_ensemble
        self.n_parameter_samples = n_parameter_samples
        self.threshold = threshold
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.time_limit = time_limit
        self.random_state = random_state

        self.classifiers = []
        self.weights = []
        self.weight_sum = 0
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
        y), either through randomising over the para
         space to make a fixed size ensemble quickly or by creating a
         variable size ensemble of those within a threshold
         of the best
        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y, enforce_univariate=True)
        y = y.values if isinstance(y, pd.Series) else y

        self.time_limit = self.time_limit * 60
        self.n_instances, self.series_length = X.shape[0], len(X.iloc[0, 0])
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifiers = []
        self.weights = []

        # Window length parameter space dependent on series length

        max_window_searches = self.series_length / 4
        max_window = int(self.series_length * self.max_win_len_prop)
        win_inc = int((max_window - self.min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1

        # cBOSS
        if self.randomised_ensemble:
            possible_parameters = self._unique_parameters(max_window, win_inc)
            num_classifiers = 0
            start_time = time.time()
            train_time = 0
            subsample_size = int(self.n_instances * 0.7)
            lowest_acc = 0
            lowest_acc_idx = 0

            rng = check_random_state(self.random_state)

            if self.time_limit > 0:
                self.n_parameter_samples = 0

            while (train_time < self.time_limit or num_classifiers <
                   self.n_parameter_samples) and len(possible_parameters) > 0:
                parameters = possible_parameters.pop(
                    rng.randint(0, len(possible_parameters)))

                subsample = rng.choice(self.n_instances, size=subsample_size,
                                       replace=False)
                X_subsample = X.iloc[subsample, :]
                y_subsample = y[subsample]

                boss = BOSSIndividual(*parameters,
                                      alphabet_size=self.alphabet_size,
                                      save_words=False,
                                      random_state=self.random_state)
                boss.fit(X_subsample, y_subsample)
                boss._clean()

                boss.accuracy = self._individual_train_acc(boss, y_subsample,
                                                           subsample_size,
                                                           lowest_acc)
                weight = math.pow(boss.accuracy, 4)

                if num_classifiers < self.max_ensemble_size:
                    if boss.accuracy < lowest_acc:
                        lowest_acc = boss.accuracy
                        lowest_acc_idx = num_classifiers
                    self.weights.append(weight)
                    self.classifiers.append(boss)

                elif boss.accuracy > lowest_acc:
                    self.weights[lowest_acc_idx] = weight
                    self.classifiers[lowest_acc_idx] = boss
                    lowest_acc, lowest_acc_idx = self._worst_ensemble_acc()

                num_classifiers += 1
                train_time = time.time() - start_time
        # BOSS
        else:
            max_acc = -1
            min_max_acc = -1

            for i, normalise in enumerate(self.norm_options):
                for win_size in range(self.min_window, max_window + 1,
                                      win_inc):
                    boss = BOSSIndividual(win_size, self.word_lengths[0],
                                          normalise, self.alphabet_size,
                                          save_words=True,
                                          random_state=self.random_state)
                    boss.fit(X, y)

                    best_classifier_for_win_size = boss
                    best_acc_for_win_size = -1

                    # the used work length may be shorter
                    best_word_len = boss.transformer.word_length

                    for n, word_len in enumerate(self.word_lengths):
                        if n > 0:
                            boss = boss._shorten_bags(word_len)

                        boss.accuracy = self._individual_train_acc(
                            boss, y, self.n_instances, best_acc_for_win_size)

                        # print(win_size, boss.accuracy)
                        if boss.accuracy >= best_acc_for_win_size:
                            best_acc_for_win_size = boss.accuracy
                            best_classifier_for_win_size = boss
                            best_word_len = word_len

                    if self._include_in_ensemble(best_acc_for_win_size,
                                                 max_acc,
                                                 min_max_acc,
                                                 len(self.classifiers)):
                        best_classifier_for_win_size._clean()
                        best_classifier_for_win_size._set_word_len(
                            best_word_len)
                        self.classifiers.append(best_classifier_for_win_size)

                        # print("appending", best_acc_for_win_size, win_size)
                        if best_acc_for_win_size > max_acc:
                            max_acc = best_acc_for_win_size
                            self.classifiers = list(compress(
                                self.classifiers, [
                                    classifier.accuracy >= max_acc *
                                    self.threshold for c, classifier in
                                    enumerate(self.classifiers)]))

                        min_max_acc, min_acc_ind = \
                            self._worst_ensemble_acc()

                        if len(self.classifiers) > self.max_ensemble_size:
                            if min_acc_ind > -1:
                                del self.classifiers[min_acc_ind]
                                min_max_acc, min_acc_ind = \
                                    self._worst_ensemble_acc()

            self.weights = [1 for n in range(len(self.classifiers))]

        self.n_estimators = len(self.classifiers)
        self.weight_sum = np.sum(self.weights)

        self._is_fitted = True
        return self

    def predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array([self.classes_[int(rng.choice(
            np.flatnonzero(prob == prob.max())))] for prob
                in self.predict_proba(X)])

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        sums = np.zeros((X.shape[0], self.n_classes))

        for n, clf in enumerate(self.classifiers):
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self.class_dictionary[preds[i]]] += self.weights[n]

        dists = sums / (np.ones(self.n_classes) * self.weight_sum)

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
        divisor = (np.ones(self.n_classes) * np.sum(self.weights))
        for i in range(num_inst):
            sums = np.zeros(self.n_classes)

            for n, clf in enumerate(self.classifiers):
                sums[self.class_dictionary.get(clf._train_predict(i), -1)] += \
                    self.weights[n]

            dists = sums / divisor
            for n in range(self.n_classes):
                results[i][n] = dists[n]

        return results

    def _unique_parameters(self, max_window, win_inc):
        possible_parameters = [[win_size, word_len, normalise] for n, normalise
                               in enumerate(self.norm_options)
                               for win_size in
                               range(self.min_window, max_window + 1, win_inc)
                               for g, word_len in enumerate(self.word_lengths)]

        return possible_parameters

    def _individual_train_acc(self, boss, y, train_size, lowest_acc):
        correct = 0
        required_correct = int(lowest_acc * train_size)

        for i in range(train_size):
            if correct + train_size - i < required_correct:
                return -1

            c = boss._train_predict(i)

            if c == y[i]:
                correct += 1

        return correct / train_size


class BOSSIndividual(BaseClassifier):
    """ Single Bag of SFA Symbols (BOSS) classifier

    Bag of SFA Symbols Ensemble: implementation of BOSS from Schaffer :
    @article
    """

    def __init__(self,
                 window_size=10,
                 word_length=8,
                 norm=False,
                 alphabet_size=4,
                 save_words=True,
                 random_state=None
                 ):
        self.window_size = window_size
        self.word_length = word_length
        self.norm = norm
        self.alphabet_size = alphabet_size

        self.save_words = save_words
        self.random_state = random_state

        self.transformer = SFA(word_length=word_length,
                               alphabet_size=alphabet_size,
                               window_size=window_size, norm=norm,
                               remove_repeat_words=True,
                               bigrams=False,
                               save_words=save_words)
        self.transformed_data = []
        self.accuracy = 0

        self.class_vals = []
        self.num_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        super(BOSSIndividual, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, enforce_univariate=True)

        sfa = self.transformer.fit_transform(X)
        self.transformed_data = sfa.iloc[:, 0]

        self.class_vals = y
        self.num_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self._is_fitted = True
        return self

    def predict(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        rng = check_random_state(self.random_state)

        classes = []
        test_bags = self.transformer.transform(X)
        test_bags = test_bags.iloc[:, 0]

        for i, test_bag in enumerate(test_bags):
            best_dist = sys.float_info.max
            nn = None

            for n, bag in enumerate(self.transformed_data):
                dist = boss_distance(test_bag, bag, best_dist)

                if dist < best_dist or (dist == best_dist and rng.random()
                                        < 0.5):
                    best_dist = dist
                    nn = self.class_vals[n]

            classes.append(nn)

        return np.array(classes)

    def predict_proba(self, X):
        preds = self.predict(X)
        dists = np.zeros((X.shape[0], self.num_classes))

        for i in range(0, X.shape[0]):
            dists[i, self.class_dictionary.get(preds[i])] += 1

        return dists

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
        new_boss = BOSSIndividual(self.window_size, word_len,
                                  self.norm, self.alphabet_size,
                                  save_words=self.save_words,
                                  random_state=self.random_state)
        new_boss.transformer = self.transformer
        sfa = self.transformer._shorten_bags(word_len)
        new_boss.transformed_data = sfa.iloc[:, 0]

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
            buf = (val_a - val_b)
            dist += buf * buf

            if dist > best_dist:
                return sys.float_info.max
    else:
        dist = np.sum([0 if first[n] == 0 else (first[n] - second[n]) * (
                first[n] - second[n])
                       for n in range(len(first))])

    return dist
