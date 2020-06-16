""" TDE classifiers
dictionary based TDE classifiers based on SFA transform. Contains a single
IndividualTDE and TDE.
"""

__author__ = "Matthew Middlehurst"
__all__ = ["TDE", "IndividualTDE", "histogram_intersection"]

import math
import random
import sys
import time

import numpy as np
from sklearn.utils.multiclass import class_distribution
from sklearn.gaussian_process import GaussianProcessRegressor
from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y


class TDE(BaseClassifier):
    """ Temporal Dictionary Ensemble (TDE)

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
    each with a LOOCV. If then retains
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
    /timeseriesweka/classifiers/dictionary_based/BOSS.java


    Parameters
    ----------
    randomised_ensemble     : bool, turns the option to just randomise the
    ensemble members rather than cross validate (cBOSS) (default=False)
    n_parameter_samples     : int, if search is randomised, number of
    parameter combos to try
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)
    threshold               : double [0,1]. retain all classifiers within
    threshold% of the best one, optional (default =0.92)
    max_ensemble_size       : int or None, retain a maximum number of
    classifiers, even if within threshold, optional
    (default = 500, recommended 50 for cBOSS)
    alphabet_size           : range of alphabet sizes to try (default to
    single value, 4)
    max_win_len_prop        : maximum window length as a proportion of
    series length (default =1)
    time_limit              : time contract to limit build time in minutes (
    default=0, no limit)
    alphabet_size           : range of alphabet size to search for (default,
    a single value a=4)
    min_window              : minimum window size, (default=10)

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
                 n_parameter_samples=250,
                 max_ensemble_size=100,
                 time_limit=0.0,
                 max_win_len_prop=1,
                 min_window=10,
                 randomly_selected_params=50,
                 random_state=None
                 ):
        self.n_parameter_samples = n_parameter_samples
        self.random_state = random_state
        random.seed(random_state)
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.time_limit = time_limit
        self.randomly_selected_params = randomly_selected_params

        self.classifiers = []
        self.weights = []
        self.weight_sum = 0
        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        self.n_estimators = 0
        self.series_length = 0
        self.n_instances = 0
        self.prev_parameters_x = []
        self.prev_parameters_y = []

        self.word_lengths = [16, 14, 12, 10, 8]
        self.norm_options = [True, False]
        self.alphabet_size = 4
        self.min_window = min_window
        self.levels = [1, 2, 3]
        self.igb_options = [True, False]
        super(TDE, self).__init__()

    def fit(self, X, y):
        """Build an ensemble of BOSS classifiers from the training set (X,
        y), either through randomising over the para
         space to make a fixed size ensemble quickly or by creating a
         variable size ensemble of those within a threshold
         of the best
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, series_length]
            The training input samples.  If a Pandas data frame is passed,
            it must have a single column. BOSS not configured
            to handle multivariate
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y, enforce_univariate=True)

        self.time_limit = self.time_limit * 6e+10
        self.n_instances, self.series_length = X.shape[0], len(X.iloc[0, 0])
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        # Window length parameter space dependent on series length
        max_window_searches = self.series_length / 4
        max_window = int(self.series_length * self.max_win_len_prop)
        win_inc = int((max_window - self.min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1

        possible_parameters = self._unique_parameters(max_window, win_inc)
        num_classifiers = 0
        start_time = time.time_ns()
        train_time = 0
        subsample_size = int(self.n_instances * 0.7)
        lowest_acc = 0
        lowest_acc_idx = 0

        if self.time_limit > 0:
            self.n_parameter_samples = 0

        while (train_time < self.time_limit or num_classifiers <
               self.n_parameter_samples) and len(possible_parameters) > 0:
            if num_classifiers < self.randomly_selected_params:
                parameters = possible_parameters.pop(
                    random.randint(0, len(possible_parameters) - 1))
            else:
                gp = GaussianProcessRegressor()
                gp.fit(self.prev_parameters_x, self.prev_parameters_y)
                preds = gp.predict(possible_parameters)
                parameters = possible_parameters.pop(np.random.choice(
                    np.flatnonzero(preds == preds.max())))

            subsample = np.random.randint(self.n_instances,
                                          size=subsample_size)
            X_subsample = X.iloc[subsample, :]
            y_subsample = y[subsample]

            tde = IndividualTDE(parameters[0], parameters[1],
                                self.alphabet_size, parameters[2],
                                parameters[3], parameters[4])
            tde.fit(X_subsample, y_subsample)

            tde.accuracy = self._individual_train_acc(tde, y_subsample,
                                                      subsample_size,
                                                      lowest_acc)
            weight = math.pow(tde.accuracy, 4)

            if num_classifiers < self.max_ensemble_size:
                if tde.accuracy < lowest_acc:
                    lowest_acc = tde.accuracy
                    lowest_acc_idx = num_classifiers
                self.weights.append(weight)
                self.classifiers.append(tde)
            elif tde.accuracy > lowest_acc:
                self.weights[lowest_acc_idx] = weight
                self.classifiers[lowest_acc_idx] = tde
                lowest_acc, lowest_acc_idx = self._worst_ensemble_acc()

            self.prev_parameters_x.append(parameters)
            self.prev_parameters_y.append(tde.accuracy)

            num_classifiers += 1
            train_time = time.time_ns() - start_time

        self.n_estimators = len(self.classifiers)
        self.weight_sum = np.sum(self.weights)

        self._is_fitted = True
        return self

    def predict(self, X):
        return [self.classes_[int(np.random.choice(
            np.flatnonzero(prob == prob.max())))] for prob
                in self.predict_proba(X)]

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        sums = np.zeros((X.shape[0], self.n_classes))

        for n, clf in enumerate(self.classifiers):
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self.class_dictionary.get(preds[i])] += self.weights[n]

        dists = sums / (np.ones(self.n_classes) * self.weight_sum)

        return dists

    def _worst_ensemble_acc(self):
        min_acc = -1
        min_acc_idx = 0

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
        possible_parameters = [[win_size, word_len, normalise, levels, igb]
                               for n, normalise in enumerate(self.norm_options)
                               for win_size in
                               range(self.min_window, max_window + 1, win_inc)
                               for w, word_len in enumerate(self.word_lengths)
                               for le, levels in enumerate(self.levels)
                               for i, igb in enumerate(self.igb_options)]

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


class IndividualTDE(BaseClassifier):
    """ Single Bag of SFA Symbols (BOSS) classifier

    Bag of SFA Symbols Ensemble: implementation of BOSS from Schaffer :
    @article
    """

    def __init__(self,
                 window_size=10,
                 word_length=8,
                 alphabet_size=4,
                 norm=False,
                 levels=1,
                 igb=False
                 ):
        self.window_size = window_size
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.norm = norm
        self.levels = levels
        self.igb = igb

        self.transform = SFA(word_length, alphabet_size,
                             window_size=window_size, norm=norm,
                             levels=levels, igb=igb, bigrams=True,
                             remove_repeat_words=True,
                             save_words=False)
        self.transformed_data = []
        self.accuracy = 0

        self.class_vals = []
        self.num_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        super(IndividualTDE, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, enforce_univariate=True)

        sfa = self.transform.fit_transform(X, y)
        self.transformed_data = [series.to_dict() for series in sfa.iloc[:, 0]]

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

        num_insts = X.shape[0]
        classes = np.zeros(num_insts, dtype=np.int_)

        test_bags = self.transform.transform(X)
        test_bags = [series.to_dict() for series in test_bags.iloc[:, 0]]

        for i, test_bag in enumerate(test_bags):
            best_sim = sys.float_info.min
            nn = -1

            for n, bag in enumerate(self.transformed_data):
                sim = histogram_intersection(test_bag, bag)

                if sim > best_sim or (sim == best_sim and
                                      random.random() < 0.5):
                    best_sim = sim
                    nn = self.class_vals[n]

            classes[i] = nn

        return classes

    def predict_proba(self, X):
        preds = self.predict(X)
        dists = np.zeros((X.shape[0], self.num_classes))

        for i in range(0, X.shape[0]):
            dists[i, self.class_dictionary.get(preds[i])] += 1

        return dists

    def _train_predict(self, train_num):
        test_bag = self.transformed_data[train_num]
        best_sim = sys.float_info.min
        nn = -1

        for n, bag in enumerate(self.transformed_data):
            if n == train_num:
                continue

            sim = histogram_intersection(test_bag, bag)

            if sim > best_sim:
                best_sim = sim
                nn = self.class_vals[n]

        return nn


def histogram_intersection(first, second):
    sim = 0

    if isinstance(first, dict):
        for word, val_a in first.items():
            val_b = second.get(word, 0)
            sim += min(val_a, val_b)
    else:
        sim = np.sum([0 if first[n] == 0 else np.min(first[n], second[n])
                      for n in range(len(first))])

    return sim
