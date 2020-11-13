# -*- coding: utf-8 -*-
""" TDE classifiers
dictionary based TDE classifiers based on SFA transform. Contains a single
IndividualTDE and TDE.
"""

__author__ = "Matthew Middlehurst"
__all__ = ["TemporalDictionaryEnsemble", "IndividualTDE", "histogram_intersection"]

import math
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformers.panel.dictionary_based import SFA
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class TemporalDictionaryEnsemble(BaseClassifier):
    """Temporal Dictionary Ensemble (TDE)

    @inproceedings{middlehurst2020temporal,
      title={The Temporal Dictionary Ensemble {(TDE)} Classifier
             for Time Series Classification},
      author={Middlehurst, Matthew and Large, James and
              Cawley, Gavin and Bagnall, Anthony},
      booktitle={The European Conference on Machine Learning and
                 Principles and Practice of Knowledge Discovery in
                 Databases},
      year={2020}
    }
    https://ueaeprints.uea.ac.uk/id/eprint/75490/

    Overview: Input n series length m
    TDE searches k parameter values selected using a Gaussian processes
    regressor, evaluating each with a LOOCV. It then retains s
    ensemble members.
    There are six primary parameters for individual classifiers:
            alpha: alphabet size
            w: window length
            l: word length
            p: normalise/no normalise
            h: levels
            b: MCB/IGB
    for any combination, an individual TDE classifier slides a window of
    length w along the series. The w length window is shortened to
    an l length word through taking a Fourier transform and keeping the
    first l/2 complex coefficients. These l
    coefficients are then discretised into alpha possible values, to form a
    word length l using breakpoints found using b. A histogram of words for
    each series is formed and stored, using a spatial pyramid of h levels.
    fit involves finding n histograms.

    predict uses 1 nearest neighbour with a the histogram intersection
    distance function.

    For the original Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/dictionary_based/TDE.java


    Parameters
    ----------
    n_parameter_samples     : int, if search is randomised, number of
    parameter combos to try (default=250)
    max_ensemble_size       : int or None, retain a maximum number of
    classifiers, even if within threshold, optional (default = 100)
    time_limit              : time contract to limit build time in minutes
    (default=0, no limit)
    max_win_len_prop        : maximum window length as a proportion of
    series length (default =1)
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
    prev_parameters_x       : parameter value of previous classifiers for GP
    prev_parameters_y       : accuracy of previous classifiers for GP

    """

    def __init__(
        self,
        n_parameter_samples=250,
        max_ensemble_size=50,
        time_limit=0.0,
        max_win_len_prop=1,
        min_window=10,
        randomly_selected_params=50,
        random_state=None,
    ):
        self.n_parameter_samples = n_parameter_samples
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.time_limit = time_limit
        self.randomly_selected_params = randomly_selected_params
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
        self.prev_parameters_x = []
        self.prev_parameters_y = []

        self.word_lengths = [16, 14, 12, 10, 8]
        self.norm_options = [True, False]
        self.min_window = min_window
        self.levels = [1, 2, 3]
        self.igb_options = [True, False]
        self.alphabet_size = 4
        super(TemporalDictionaryEnsemble, self).__init__()

    def fit(self, X, y):
        """Build an ensemble of individual TDE classifiers from the training
        set (X,y), through randomising over the parameter space to a set
        number of times then selecting new parameters using Gaussian
        processes

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)

        self.time_limit = self.time_limit * 60
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifiers = []
        self.weights = []
        self.prev_parameters_x = []
        self.prev_parameters_y = []

        # Window length parameter space dependent on series length
        max_window_searches = self.series_length / 4
        max_window = int(self.series_length * self.max_win_len_prop)
        win_inc = int((max_window - self.min_window) / max_window_searches)
        if win_inc < 1:
            win_inc = 1

        possible_parameters = self._unique_parameters(max_window, win_inc)
        num_classifiers = 0
        start_time = time.time()
        train_time = 0
        subsample_size = int(self.n_instances * 0.7)
        lowest_acc = 1
        lowest_acc_idx = 0

        if self.time_limit > 0:
            self.n_parameter_samples = 0

        rng = check_random_state(self.random_state)

        while (
            train_time < self.time_limit or num_classifiers < self.n_parameter_samples
        ) and len(possible_parameters) > 0:
            if num_classifiers < self.randomly_selected_params:
                parameters = possible_parameters.pop(
                    rng.randint(0, len(possible_parameters))
                )
            else:
                gp = GaussianProcessRegressor(random_state=self.random_state)
                gp.fit(self.prev_parameters_x, self.prev_parameters_y)
                preds = gp.predict(possible_parameters)
                parameters = possible_parameters.pop(
                    rng.choice(np.flatnonzero(preds == preds.max()))
                )

            subsample = rng.choice(self.n_instances, size=subsample_size, replace=False)
            X_subsample = X[subsample]
            y_subsample = y[subsample]

            tde = IndividualTDE(
                *parameters,
                alphabet_size=self.alphabet_size,
                random_state=self.random_state
            )
            tde.fit(X_subsample, y_subsample)

            tde.accuracy = self._individual_train_acc(
                tde, y_subsample, subsample_size, lowest_acc
            )
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
            train_time = time.time() - start_time

        self.n_estimators = len(self.classifiers)
        self.weight_sum = np.sum(self.weights)

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

        for n, clf in enumerate(self.classifiers):
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self.class_dictionary[preds[i]]] += self.weights[n]

        dists = sums / (np.ones(self.n_classes) * self.weight_sum)

        return dists

    def _worst_ensemble_acc(self):
        min_acc = 1.0
        min_acc_idx = 0

        for c, classifier in enumerate(self.classifiers):
            if classifier.accuracy < min_acc:
                min_acc = classifier.accuracy
                min_acc_idx = c

        return min_acc, min_acc_idx

    def _get_train_probs(self, X):
        num_inst = X.shape[0]
        results = np.zeros((num_inst, self.n_classes))
        divisor = np.ones(self.n_classes) * np.sum(self.weights)
        for i in range(num_inst):
            sums = np.zeros(self.n_classes)

            for n, clf in enumerate(self.classifiers):
                sums[
                    self.class_dictionary.get(clf._train_predict(i), -1)
                ] += self.weights[n]

            dists = sums / divisor
            for n in range(self.n_classes):
                results[i][n] = dists[n]

        return results

    def _unique_parameters(self, max_window, win_inc):
        possible_parameters = [
            [win_size, word_len, normalise, levels, igb]
            for n, normalise in enumerate(self.norm_options)
            for win_size in range(self.min_window, max_window + 1, win_inc)
            for w, word_len in enumerate(self.word_lengths)
            for le, levels in enumerate(self.levels)
            for i, igb in enumerate(self.igb_options)
        ]

        return possible_parameters

    def _individual_train_acc(self, tde, y, train_size, lowest_acc):
        correct = 0
        required_correct = int(lowest_acc * train_size)

        for i in range(train_size):
            if correct + train_size - i < required_correct:
                return -1

            c = tde._train_predict(i)

            if c == y[i]:
                correct += 1

        return correct / train_size


class IndividualTDE(BaseClassifier):
    """Single TDE classifier, based off the Bag of SFA Symbols (BOSS) model"""

    def __init__(
        self,
        window_size=10,
        word_length=8,
        norm=False,
        levels=1,
        igb=False,
        alphabet_size=4,
        random_state=None,
    ):
        self.window_size = window_size
        self.word_length = word_length
        self.norm = norm
        self.levels = levels
        self.igb = igb
        self.alphabet_size = alphabet_size

        self.random_state = random_state

        binning_method = "information-gain" if igb else "equi-depth"

        self.transformer = SFA(
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=window_size,
            norm=norm,
            levels=levels,
            binning_method=binning_method,
            bigrams=True,
            remove_repeat_words=True,
            save_words=False,
        )
        self.transformed_data = []
        self.accuracy = 0

        self.class_vals = []
        self.num_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        super(IndividualTDE, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)

        sfa = self.transformer.fit_transform(X, y)
        self.transformed_data = sfa[0]  # .iloc[:, 0]

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

        rng = check_random_state(self.random_state)

        classes = []
        test_bags = self.transformer.transform(X)
        test_bags = test_bags[0]  # .iloc[:, 0]

        for test_bag in test_bags:
            best_sim = -1
            nn = None

            for n, bag in enumerate(self.transformed_data):
                sim = histogram_intersection(test_bag, bag)

                if sim > best_sim or (sim == best_sim and rng.random() < 0.5):
                    best_sim = sim
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
        best_sim = -1
        nn = None

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
        sim = np.sum(
            [
                0 if first[n] == 0 else np.min(first[n], second[n])
                for n in range(len(first))
            ]
        )

    return sim
