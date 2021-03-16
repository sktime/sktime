# -*- coding: utf-8 -*-
""" TDE classifiers
dictionary based TDE classifiers based on SFA transform. Contains a single
IndividualTDE and TDE.
"""

__author__ = "Matthew Middlehurst"
__all__ = ["TemporalDictionaryEnsemble", "IndividualTDE", "histogram_intersection"]

import math
import time
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class TemporalDictionaryEnsemble(BaseClassifier):
    """
    Temporal Dictionary Ensemble (TDE) as described in [1].

    Overview: Input n series length m with d dimensions
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
    first l/2 complex coefficients. These lcoefficients are then discretised
    into alpha possible values, to form a word length l using breakpoints
    found using b. A histogram of words for each series is formed and stored,
    using a spatial pyramid of h levels. For multivariate series, accuracy
    from a reduced histogram is used to select dimensions.

    fit involves finding n histograms.
    predict uses 1 nearest neighbour with a the histogram intersection
    distance function.

    For the original Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/TDE.java


    Parameters
    ----------
    n_parameter_samples     : int, number of parameter combos to try
    (default=250)
    max_ensemble_size       : int, maximum number of classifiers
    (default=50)
    time_limit              : int, time contract to limit build time in
    minutes (default=0, no limit)
    max_win_len_prop        : float between 0 and 1, maximum window length
    as a proportion of series length (default=1)
    min_window              : int, minimum window size (default=10)
    randomly_selected_params: int, number of parameters randomly selected
    before GP is used (default=50)
    bigrams                 : boolean or None, whether to use bigrams
    (default=None, true for univariate, false for multivariate)
    dim_threshold           : float between 0 and 1, dimension accuracy
    threshold for multivariate (default=0.85)
    max_dims                : int, max number of dimensions for multivariate
    (default=20)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    n_classes               : extracted from the data
    n_instances             : extracted from the data
    n_dims                  : extracted from the data
    n_estimators            : The final number of classifiers used
    (<=max_ensemble_size)
    series_length           : length of all series (assumed equal)
    classifiers             : array of IndividualTDE classifiers
    weights                 : weight of each classifier in the ensemble
    weight_sum              : sum of all weights
    prev_parameters_x       : parameter value of previous classifiers for GP
    prev_parameters_y       : accuracy of previous classifiers for GP

    Notes
    -----
    ..[1] Matthew Middlehurst, James Large, Gavin Cawley and Anthony Bagnall
        "The Temporal Dictionary Ensemble (TDE) Classifier for Time Series
        Classification",
            in proceedings of the European Conference on Machine Learning and
            Principles and Practice of Knowledge Discovery in Databases, 2020
        https://ueaeprints.uea.ac.uk/id/eprint/75490/

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/dictionary_based/TDE.java

    """

    # Capabilities: data types this classifier can handle
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        n_parameter_samples=250,
        max_ensemble_size=50,
        time_limit=0.0,
        max_win_len_prop=1,
        min_window=10,
        randomly_selected_params=50,
        bigrams=None,
        dim_threshold=0.85,
        max_dims=20,
        n_jobs=1,
        random_state=None,
    ):
        self.n_parameter_samples = n_parameter_samples
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.time_limit = time_limit
        self.randomly_selected_params = randomly_selected_params
        self.bigrams = bigrams

        self.n_jobs = n_jobs
        self.random_state = random_state

        # multivariate
        self.dim_threshold = dim_threshold
        self.max_dims = max_dims

        self.classifiers = []
        self.weights = []
        self.weight_sum = 0
        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        self.n_estimators = 0
        self.series_length = 0
        self.n_dims = 0
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
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        self.time_limit = self.time_limit * 60
        self.n_instances, self.n_dims, self.series_length = X.shape
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
        if self.min_window > max_window + 1:
            raise ValueError(
                f"Error in TemporalDictionaryEnsemble, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={self.max_window},"
                f" series length is {self.series_length}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )
        rng = check_random_state(self.random_state)

        if self.bigrams is None:
            if self.n_dims > 1:
                use_bigrams = False
            else:
                use_bigrams = True
        else:
            use_bigrams = self.bigrams

        # use time limit or n_parameter_samples if limit is 0
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
                bigrams=use_bigrams,
                dim_threshold=self.dim_threshold,
                max_dims=self.max_dims,
                random_state=self.random_state,
            )
            tde.fit(X_subsample, y_subsample)
            tde.subsample = subsample

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
        X = check_X(X, coerce_to_numpy=True)

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

    def _get_train_probs(self, X):
        num_inst = X.shape[0]
        results = np.zeros((num_inst, self.n_classes))
        for i in range(num_inst):
            divisor = 0
            sums = np.zeros(self.n_classes)

            cls_idx = []
            for n, clf in enumerate(self.classifiers):
                idx = np.where(clf.subsample == i)
                if len(idx[0]) > 0:
                    cls_idx.append([n, idx[0][0]])

            preds = Parallel(n_jobs=self.n_jobs)(
                delayed(self.classifiers[cls[0]]._train_predict)(
                    cls[1],
                )
                for cls in cls_idx
            )

            for n, pred in enumerate(preds):
                sums[self.class_dictionary.get(pred, -1)] += self.weights[cls_idx[n][0]]
                divisor += self.weights[cls_idx[n][0]]

            results[i] = (
                np.ones(self.n_classes) * (1 / self.n_classes)
                if divisor == 0
                else sums / (np.ones(self.n_classes) * divisor)
            )

        return results

    def _individual_train_acc(self, tde, y, train_size, lowest_acc):
        correct = 0
        required_correct = int(lowest_acc * train_size)

        if self.n_jobs > 1:
            c = Parallel(n_jobs=self.n_jobs)(
                delayed(tde._train_predict)(
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
        bigrams=True,
        dim_threshold=0.85,
        max_dims=20,
        n_jobs=1,
        random_state=None,
    ):
        self.window_size = window_size
        self.word_length = word_length
        self.norm = norm
        self.levels = levels
        self.igb = igb
        self.alphabet_size = alphabet_size
        self.bigrams = bigrams

        # multivariate
        self.dim_threshold = dim_threshold
        self.max_dims = max_dims

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.transformers = []
        self.transformed_data = []
        self.accuracy = 0
        self.subsample = []

        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.highest_dim_bit = 0
        self.dims = []
        self.class_vals = []
        self.num_classes = 0
        self.classes_ = []
        self.class_dictionary = {}
        super(IndividualTDE, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        self.n_instances, self.n_dims, self.series_length = X.shape
        self.class_vals = y
        self.num_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        # select dimensions using accuracy estimate if multivariate
        if self.n_dims > 1:
            self.dims, self.transformers = self._select_dims(X, y)

            words = [defaultdict(int) for _ in range(self.n_instances)]

            for i, dim in enumerate(self.dims):
                X_dim = X[:, dim, :].reshape(self.n_instances, 1, self.series_length)
                dim_words = self.transformers[i].transform(X_dim, y)
                dim_words = dim_words[0]

                for i in range(self.n_instances):
                    for word, count in dim_words[i].items():
                        words[i][word << self.highest_dim_bit | dim] = count

            self.transformed_data = words
        else:
            self.transformers.append(
                SFA(
                    word_length=self.word_length,
                    alphabet_size=self.alphabet_size,
                    window_size=self.window_size,
                    norm=self.norm,
                    levels=self.levels,
                    binning_method="information-gain" if self.igb else "equi-depth",
                    bigrams=self.bigrams,
                    remove_repeat_words=True,
                    save_words=False,
                    n_jobs=self.n_jobs,
                )
            )
            sfa = self.transformers[0].fit_transform(X, y)
            self.transformed_data = sfa[0]

        self._is_fitted = True
        return self

    def predict(self, X):
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)
        num_cases = X.shape[0]

        if self.n_dims > 1:
            words = [defaultdict(int) for _ in range(num_cases)]

            for i, dim in enumerate(self.dims):
                X_dim = X[:, dim, :].reshape(num_cases, 1, self.series_length)
                dim_words = self.transformers[i].transform(X_dim)
                dim_words = dim_words[0]

                for i in range(num_cases):
                    for word, count in dim_words[i].items():
                        words[i][word << self.highest_dim_bit | dim] = count

            test_bags = words
        else:
            test_bags = self.transformers[0].transform(X)
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

        best_sim = -1
        nn = None

        for n, bag in enumerate(self.transformed_data):
            sim = histogram_intersection(test_bag, bag)

            if sim > best_sim or (sim == best_sim and rng.random() < 0.5):
                best_sim = sim
                nn = self.class_vals[n]

        return nn

    def _select_dims(self, X, y):
        self.highest_dim_bit = (math.ceil(math.log2(self.n_dims))) + 1
        accs = []
        transformers = []

        # select dimensions based on reduced bag size accuracy
        for i in range(self.n_dims):
            self.dims.append(i)
            transformers.append(
                SFA(
                    word_length=self.word_length,
                    alphabet_size=self.alphabet_size,
                    window_size=self.window_size,
                    norm=self.norm,
                    levels=self.levels,
                    binning_method="information-gain" if self.igb else "equi-depth",
                    bigrams=self.bigrams,
                    remove_repeat_words=True,
                    save_words=False,
                    save_binning_dft=True,
                    n_jobs=self.n_jobs,
                )
            )

            X_dim = X[:, i, :].reshape(self.n_instances, 1, self.series_length)

            transformers[i].fit(X_dim, y)
            sfa = transformers[i].transform(
                X_dim,
                y,
                transformers[i].binning_dft,
            )
            transformers[i].binning_dft = None

            correct = 0
            for i in range(self.n_instances):
                if self._train_predict(i, sfa[0]) == y[i]:
                    correct = correct + 1

            accs.append(correct)

        max_acc = max(accs)

        dims = []
        fin_transformers = []
        for i in range(self.n_dims):
            if accs[i] >= max_acc * self.dim_threshold:
                dims.append(i)
                fin_transformers.append(transformers[i])

        if len(dims) > self.max_dims:
            idx = self.random_state.choice(
                len(dims),
                self.max_dims,
                replace=False,
            ).tolist()
            dims = [dims[i] for i in idx]
            fin_transformers = [fin_transformers[i] for i in idx]

        return dims, fin_transformers

    def _train_predict(self, train_num, bags=None):
        if bags is None:
            bags = self.transformed_data

        test_bag = bags[train_num]
        best_sim = -1
        nn = None

        for n, bag in enumerate(bags):
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
