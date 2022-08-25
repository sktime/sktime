# -*- coding: utf-8 -*-
"""WEASEL on Steroids classifier.

Dictionary based classifier based on SFA transform, BOSS and linear regression.

The fastest WEASEL on earth
Tour de WEASEL
WEASEL on Steroids
WEASEL + Rocket-Science

Rocket-Science on WEASEL
"""

__author__ = ["patrickzib"]
__all__ = ["WEASEL_STEROIDS"]

from warnings import simplefilter

import numpy as np
from joblib import Parallel, delayed
from numba import NumbaPendingDeprecationWarning, njit
from numba.core import types
from numba.typed import Dict
from sklearn.feature_selection import chi2
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SAX_NEW, SFA_NEW

# TODO change to transformer?

simplefilter(action="ignore", category=NumbaPendingDeprecationWarning)


class WEASEL_STEROIDS(BaseClassifier):
    """Word Extraction for Time Series Classification (WEASEL).

    Overview: Input n series length m
    WEASEL is a dictionary classifier that builds a bag-of-patterns using SFA
    for different window lengths and learns a logistic regression classifier
    on this bag.

    There are these primary parameters:
            alphabet_size: alphabet size
            chi2-threshold: used for feature selection to select best words
            anova: select best l/2 fourier coefficients other than first ones
            bigrams: using bigrams of SFA words
            binning_strategy: the binning strategy used to discretise into
                             SFA words.
    WEASEL slides a window length w along the series. The w length window
    is shortened to an l length word through taking a Fourier transform and
    keeping the best l/2 complex coefficients using an anova one-sided
    test. These l coefficients are then discretised into alpha possible
    symbols, to form a word of length l. A histogram of words for each
    series is formed and stored.
    For each window-length a bag is created and all words are joint into
    one bag-of-patterns. Words from different window-lengths are
    discriminated by different prefixes.
    fit involves training a logistic regression classifier on the single
    bag-of-patterns.

    predict uses the logistic regression classifier

    Parameters
    ----------
    anova: boolean, default=True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given
    bigrams: boolean, default=True
        whether to create bigrams of SFA words
    binning_strategies: ["equi-depth", "equi-width", "information-gain"],
    default="information-gain"
        The binning method used to derive the breakpoints.
    random_state: int or None, default=None
        Seed for random, integer

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    MUSE

    References
    ----------
    .. [1] Patrick Schäfer and Ulf Leser, "Fast and Accurate Time Series Classification
    with WEASEL", in proc ACM on Conference on Information and Knowledge Management,
    2017, https://dl.acm.org/doi/10.1145/3132847.3132980

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/dictionary_based/WEASEL.java>`_.

    Examples
    --------
    >>> from sktime.classification.dictionary_based import WEASEL_STEROIDS
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = WEASEL_STEROIDS(window_inc=4)
    >>> clf.fit(X_train, y_train)
    WEASEL2(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capabilitys:multithreading": True,
        "classifier_type": "dictionary",
    }

    def __init__(
        self,
        anova=False,
        variance=True,
        bigrams=False,
        binning_strategies=["equi-depth"],
        n_jobs=4,
        ensemble_size=50,
        max_feature_count=20_000,
        min_window=8,
        max_window=32,
        norm_options=[True, False],
        word_lengths=[6, 8],
        alphabet_sizes=[4],
        use_first_differences=[False],
        feature_selection="random",
        random_state=None,
    ):
        # currently greater values than 4 are not supported.
        self.alphabet_sizes = alphabet_sizes

        self.anova = anova
        self.variance = variance

        self.norm_options = norm_options
        self.word_lengths = word_lengths

        self.bigrams = bigrams
        self.binning_strategies = binning_strategies
        self.random_state = random_state

        self.min_window = min_window
        self.max_window = max_window
        self.ensemble_size = ensemble_size
        self.max_feature_count = max_feature_count
        self.use_first_differences = use_first_differences
        self.window_sizes = []

        self.series_length = 0
        self.n_instances = 0

        self.SFA_transformers = []
        self.relevant_features = []
        self.rel_features_counts = []
        self.clf = None
        self.n_jobs = n_jobs

        self.feature_selection = feature_selection

        # set_num_threads(n_jobs)

        super(WEASEL_STEROIDS, self).__init__()

    def _fit(self, X, y):
        """Build a WEASEL classifiers from the training set (X, y).

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
        """
        # Window length parameter space dependent on series length
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        X = X.squeeze(1)

        self.max_window = int(min(self.series_length, self.max_window))
        if self.min_window > self.max_window:
            raise ValueError(
                f"Error in WEASEL, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={self.max_window},"
                f" series length is {self.series_length}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )

        # Randomly choose window sizes
        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        n_jobs = (
            1
            if (self.feature_selection == "random" or self.feature_selection == "chi2")
            else self.n_jobs
        )
        parallel_res = Parallel(n_jobs=n_jobs, timeout=99999)(
            delayed(_parallel_fit)(
                i,
                X,
                y,
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.series_length,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.n_instances,
                self.feature_selection,
            )
            for i in range(self.ensemble_size)
        )

        self.total_features_count = 0
        sfa_words = []
        for (
            sfa_words2,
            transformer2,
            relevant_features2,
            rel_features_count2,
        ) in parallel_res:
            self.SFA_transformers.append(transformer2)
            self.relevant_features.append(relevant_features2)
            self.rel_features_counts.append(rel_features_count2)
            self.total_features_count += sfa_words2.shape[1]
            sfa_words.append(sfa_words2)

        # merging arrays from different threads
        all_words = np.concatenate(sfa_words, axis=1)
        self.clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 10), normalize=False)
        self.clf.fit(all_words, y)
        # print(f"\tCross-Validation Acc: {self.clf.best_score_}")

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
        bag = self._transform_words(X)
        return self.clf.predict(bag)

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
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        X = X.squeeze(1)

        # X = (X - X.mean(axis=-1, keepdims=True)) / (
        #         X.std(axis=-1, keepdims=True) + 1e-8
        # )

        n_jobs = (
            1
            if (self.feature_selection == "random" or self.feature_selection == "chi2")
            else self.n_jobs
        )
        parallel_res = Parallel(n_jobs=n_jobs, timeout=99999)(
            delayed(_parallel_transform_words)(
                X,
                self.SFA_transformers[i],
                self.relevant_features[i],
                self.rel_features_counts[i],
                self.feature_selection,
            )
            for i in range(self.ensemble_size)
        )

        return np.concatenate(parallel_res, axis=1)

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
        return {}


def _parallel_fit(
    i,
    X,
    y,
    window_sizes,
    alphabet_sizes,
    word_lengths,
    series_length,
    norm_options,
    use_first_differences,
    binning_strategies,
    variance,
    anova,
    bigrams,
    n_jobs,
    max_feature_count,
    ensemble_size,
    n_instances,
    feature_selection,
):
    rng = check_random_state(i)
    window_size = rng.choice(window_sizes)
    alphabet_size = rng.choice(alphabet_sizes)

    # maximize word-length
    word_length = min(window_size - 2, rng.choice(word_lengths))
    norm = rng.choice(norm_options)
    # word_length = min(window_size - (2 if norm else 0), rng.choice(word_lengths))

    dilation = max(
        1,
        np.int32(2 ** rng.uniform(0, np.log2((series_length - 1) / (window_size - 1)))),
    )
    first_difference = rng.choice(use_first_differences)
    binning_strategy = rng.choice(binning_strategies)
    # upper = rng.choice([True, False])

    # TODO count subgroups of two letters of the words?

    # which_transform = rng.uniform(0, 1)
    if True:  # which_transform > 0.33:
        transformer = SFA_NEW(
            variance=variance,
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=window_size,
            norm=norm,
            anova=anova,
            binning_method=binning_strategy,
            bigrams=bigrams,
            dilation=dilation,
            first_difference=first_difference,
            n_jobs=n_jobs,
        )
    else:
        transformer = SAX_NEW(
            word_length=word_length, alphabet_size=4, window_size=window_size
        )

    # generate SFA words on subsample
    sfa_words = transformer.fit_transform(X, y)

    # Prefilter and remove those with only one value
    all_win_words = None
    relevant_features = None

    if feature_selection == "none":
        all_win_words = create_bag_none(
            transformer.X_index, alphabet_size, n_instances, sfa_words, word_length
        )
    else:
        feature_names = create_feature_names(sfa_words)

        # Random feature selection
        if feature_selection == "random":
            feature_count = min(max_feature_count // ensemble_size, len(feature_names))
            relevant_features_idx = rng.choice(
                len(feature_names), replace=False, size=feature_count
            )
            all_win_words, relevant_features = create_bag_feature_selection(
                n_instances,
                relevant_features_idx,
                np.array(list(feature_names)),
                sfa_words,
            )

        # Chi-squared feature selection
        elif feature_selection == "chi2":
            feature_count = len(list(feature_names))
            relevant_features_idx = np.arange(feature_count, dtype=np.int32)
            all_win_words, _ = create_bag_feature_selection(
                n_instances,
                relevant_features_idx,
                np.array(list(feature_names)),
                sfa_words,
            )

            feature_count = min(max_feature_count // ensemble_size, len(feature_names))
            chi2_statistics, p = chi2(all_win_words, y)
            relevant_features_idx = np.argsort(p)[:feature_count]

            relevant_features = Dict.empty(key_type=types.int32, value_type=types.int32)
            for k, v in zip(
                np.array(list(feature_names))[relevant_features_idx],
                np.arange(len(relevant_features_idx)),
            ):
                relevant_features[k] = v

            # select subset of features
            all_win_words = all_win_words[:, relevant_features_idx]

    return (all_win_words, transformer, relevant_features, all_win_words.shape[1])


@njit(cache=True, fastmath=True)
def create_feature_names(sfa_words):
    feature_names = set()
    for t_words in sfa_words:
        for t_word in t_words:
            feature_names.add(t_word)
    return feature_names


@njit(cache=True, fastmath=True)
def create_bag_none(X_index, alphabet_size, n_instances, sfa_words, word_length):
    feature_count = np.int32(2**word_length)
    all_win_words = np.zeros((n_instances, feature_count), dtype=np.int32)

    for j in range(len(sfa_words)):
        all_win_words[j, :] = np.bincount(sfa_words[j], minlength=feature_count)

    # cc = feature_count // 2 + key
    # if all_win_words[j, cc] == 0:
    #    all_win_words[j, cc] = X_index[k] / sfa_words.shape[1]

    return all_win_words


@njit(cache=True, fastmath=True)
def create_bag_feature_selection(
    n_instances, relevant_features_idx, feature_names, sfa_words
):
    relevant_features = Dict.empty(key_type=types.int32, value_type=types.int32)
    for k, v in zip(
        feature_names[relevant_features_idx],
        np.arange(len(relevant_features_idx)),
    ):
        relevant_features[k] = v

    all_win_words = np.zeros((n_instances, len(relevant_features_idx)), dtype=np.int32)
    for j in range(len(sfa_words)):
        for key in sfa_words[j]:
            if key in relevant_features:
                all_win_words[j, relevant_features[key]] += 1
    return all_win_words, relevant_features


@njit(cache=True, fastmath=True)
def create_bag_transform(
    X, X_index, feature_count, feature_selection, relevant_features, sfa_words
):
    # merging arrays
    all_win_words = np.zeros((len(X), feature_count), np.int32)
    for j in range(len(sfa_words)):
        if feature_selection == "none":
            all_win_words[j, :] = np.bincount(sfa_words[j], minlength=feature_count)
        else:
            for _, key in enumerate(sfa_words[j]):
                if key in relevant_features:
                    o = relevant_features[key]
                    all_win_words[j, o] += 1

        # cc = feature_count // 2 + key
        # if all_win_words[j, cc] == 0:
        #    all_win_words[j, cc] = X_index[k] / sfa_words.shape[1]

    return all_win_words, feature_count


def _parallel_transform_words(
    X,
    transformer,
    relevant_features,
    feature_count,
    feature_selection,
):
    # SFA transform
    sfa_words = transformer.transform(X)

    return create_bag_transform(
        X,
        transformer.X_index,
        feature_count,
        feature_selection,
        relevant_features
        if relevant_features
        else Dict.empty(
            key_type=types.int32,
            value_type=types.int32,
        ),
        sfa_words,
    )[0]
