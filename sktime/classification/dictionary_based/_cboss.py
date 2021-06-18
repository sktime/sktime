# -*- coding: utf-8 -*-
"""ContractableBOSS classifier.

Dictionary based cBOSS classifier based on SFA transform. Improves the
ensemble structure of the original BOSS algorithm.
"""

__author__ = "Matthew Middlehurst"
__all__ = ["ContractableBOSS"]

import math
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.multiclass import class_distribution
from sklearn.utils import check_random_state
from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import IndividualBOSS
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class ContractableBOSS(BaseClassifier):
    """Contractable Bag of SFA Symbols (cBOSS).

    implementation of BOSS from [1] with refinements described in [2].

    Overview: Input n series length m
    cBOSS randomly samples n_parameter_samples parameter sets, evaluating
    each with a LOOCV. It then retains max_ensemble_size classifiers with the
    highest accuracy.
    There are three primary parameters:
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
    n_parameter_samples     : int, if search is randomised, number of
    parameter combos to try (default = 250)
    max_ensemble_size       : int or None, retain a maximum number of
    classifiers, even if within threshold, optional (default = 50)
    max_win_len_prop        : maximum window length as a proportion of
    series length (default = 1)
    time_limit_in_minutes   : time contract to limit build time in minutes
    (default = 0, no limit)
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
    weights                 : weight of each classifier in the ensemble

    See Also
    --------
    BOSSEnsemble

    Notes
    -----
    ..[1] Patrick Schäfer, "The BOSS is concerned with time series
    classification in the presence of noise",
    Data Mining and Knowledge Discovery, 29(6): 2015
            https://link.springer.com/article/10.1007/s10618-014-0377-7
    ..[2] Matthew Middlehurst, William Vickers and Anthony Bagnall
    "Scalable Dictionary Classifiers for Time Series Classification",
    in proc 20th International Conference on Intelligent Data Engineering
    and Automated Learning,LNCS, volume 11871
            https://link.springer.com/chapter/10.1007/978-3-030-33607-3_2

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/
    main/java/tsml/classifiers/dictionary_based/cBOSS.java
    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": True,
        "contractable": True,
    }

    def __init__(
        self,
        n_parameter_samples=250,
        max_ensemble_size=50,
        max_win_len_prop=1,
        min_window=10,
        time_limit_in_minutes=0.0,
        n_jobs=1,
        random_state=None,
    ):
        self.n_parameter_samples = n_parameter_samples
        self.max_ensemble_size = max_ensemble_size
        self.max_win_len_prop = max_win_len_prop
        self.time_limit_in_minutes = time_limit_in_minutes

        self.n_jobs = n_jobs
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
        super(ContractableBOSS, self).__init__()

    def fit(self, X, y):
        """Fit a c-boss ensemble on cases (X,y), where y is the target variable.

        Build an ensemble of BOSS classifiers from the training set (X,
        y), through randomising over the para space to make a fixed size
        ensemble of the best.

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

        start_time = time.time()
        time_limit = self.time_limit_in_minutes * 60
        self.n_instances, _, self.series_length = X.shape
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
        if self.min_window > max_window + 1:
            raise ValueError(
                f"Error in ContractableBOSS, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={max_window},"
                f" series length is {self.series_length}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )
        possible_parameters = self._unique_parameters(max_window, win_inc)
        num_classifiers = 0
        train_time = 0
        subsample_size = int(self.n_instances * 0.7)
        lowest_acc = 1
        lowest_acc_idx = 0

        rng = check_random_state(self.random_state)

        if time_limit > 0:
            self.n_parameter_samples = 0

        while (
            train_time < time_limit or num_classifiers < self.n_parameter_samples
        ) and len(possible_parameters) > 0:
            parameters = possible_parameters.pop(
                rng.randint(0, len(possible_parameters))
            )

            subsample = rng.choice(self.n_instances, size=subsample_size, replace=False)
            X_subsample = X[subsample]
            y_subsample = y[subsample]

            boss = IndividualBOSS(
                *parameters,
                alphabet_size=self.alphabet_size,
                save_words=False,
                random_state=self.random_state,
            )
            boss.fit(X_subsample, y_subsample)
            boss._clean()
            boss.subsample = subsample

            boss.accuracy = self._individual_train_acc(
                boss, y_subsample, subsample_size, lowest_acc
            )
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

        self.n_estimators = len(self.classifiers)
        self.weight_sum = np.sum(self.weights)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, 1]
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
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, self.n_classes]
        """
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
        min_acc_idx = -1

        for c, classifier in enumerate(self.classifiers):
            if classifier.accuracy < min_acc:
                min_acc = classifier.accuracy
                min_acc_idx = c

        return min_acc, min_acc_idx

    def _unique_parameters(self, max_window, win_inc):
        possible_parameters = [
            [win_size, word_len, normalise]
            for n, normalise in enumerate(self.norm_options)
            for win_size in range(self.min_window, max_window + 1, win_inc)
            for g, word_len in enumerate(self.word_lengths)
        ]

        return possible_parameters

    def _get_train_probs(self, X):
        num_inst = X.shape[0]
        results = np.zeros((num_inst, self.n_classes))
        for i in range(num_inst):
            divisor = 0
            sums = np.zeros(self.n_classes)

            clf_idx = []
            for n, clf in enumerate(self.classifiers):
                idx = np.where(clf.subsample == i)
                if len(idx[0]) > 0:
                    clf_idx.append([n, idx[0][0]])

            preds = Parallel(n_jobs=self.n_jobs)(
                delayed(self.classifiers[cls[0]]._train_predict)(
                    cls[1],
                )
                for cls in clf_idx
            )

            for n, pred in enumerate(preds):
                sums[self.class_dictionary.get(pred, -1)] += self.weights[clf_idx[n][0]]
                divisor += self.weights[clf_idx[n][0]]

            results[i] = (
                np.ones(self.n_classes) * (1 / self.n_classes)
                if divisor == 0
                else sums / (np.ones(self.n_classes) * divisor)
            )

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
