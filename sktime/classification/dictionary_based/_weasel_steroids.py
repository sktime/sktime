# -*- coding: utf-8 -*-
"""WEASEL on Steroids classifier.

Dictionary based classifier based on SFA transform, BOSS and linear regression.
"""

__author__ = ["patrickzib"]
__all__ = ["WEASEL_STEROIDS"]

import math

import numpy as np
from joblib import Parallel, delayed
from numba import njit

# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA


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
    binning_strategy: {"equi-depth", "equi-width", "information-gain"},
    default="information-gain"
        The binning method used to derive the breakpoints.
    window_inc: int, default=2
        WEASEL create a BoP model for each window sizes. This is the
        increment used to determine the next window size.
    p_threshold:  int, default=0.05 (disabled by default)
        Feature selection is applied based on the chi-squared test.
        This is the p-value threshold to use for chi-squared test on bag-of-words
        (lower means more strict). 1 indicates that the test
        should not be performed.
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
        "capability:multithreading": True,
        "classifier_type": "dictionary",
    }

    def __init__(
        self,
        anova=False,
        variance=False,
        bigrams=False,
        binning_strategy="equi-depth",
        window_inc=1,
        p_threshold=0.05,
        n_jobs=4,
        ensemble_size=200,
        random_state=None,
    ):

        # currently greater values than 4 are not supported.
        self.alphabet_size = 4

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold

        self.anova = anova
        self.variance = variance

        self.norm_options = [True, False]
        self.word_lengths = [6, 8]

        self.bigrams = bigrams
        self.binning_strategy = binning_strategy
        self.random_state = random_state

        self.min_window = 8
        self.max_window = 16
        self.ensemble_size = ensemble_size
        self.max_feature_count = 10_000

        self.window_inc = window_inc
        self.highest_bit = -1
        self.window_sizes = []

        self.series_length = 0
        self.n_instances = 0

        self.SFA_transformers = []
        self.dilation_factors = []
        self.relevant_features = []
        self.rel_features_counts = []
        self.clf = None
        self.n_jobs = n_jobs

        super(WEASEL_STEROIDS, self).__init__()

    @staticmethod
    @njit
    def _dilation(X, d):
        first = X[:, :, 0::d]
        for i in range(1, d):
            second = X[:, :, i::d]
            first = np.concatenate((first, second), axis=2)
        return first

    class MyDict:
        """Own Dict implementation for speed issues."""

        def __init__(self):
            self.content = []

        def update(self, bag):
            """Update the bag, adding all elements."""
            self.content.extend(bag)

        def items(self):
            """Iterate the bag."""
            return self.content

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

        win_inc = self._compute_window_inc()

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
        self.window_sizes = list(range(self.min_window, self.max_window + 1, win_inc))
        self.highest_bit = (math.ceil(math.log2(self.max_window))) + 1

        def _parallel_fit(
            i,
        ):
            rng = check_random_state(i)
            window_size = rng.choice(self.window_sizes)
            word_length = min(window_size - 2, rng.choice(self.word_lengths))

            k = 2
            dilation_sizes = np.int32(
                [2**j for j in np.arange(0, np.log2(self.series_length / k))]
            )
            dilation = rng.choice(dilation_sizes)
            norm = rng.choice(self.norm_options)

            # print(dilation_sizes, self.series_length)
            # print(window_size, word_length, dilation, norm)

            transformer = SFA(
                variance=self.variance,
                word_length=word_length,
                alphabet_size=self.alphabet_size,
                window_size=window_size,
                norm=norm,
                anova=self.anova,
                # binning_method=self.binning_strategy,
                binning_method=rng.choice(["equi-depth", "information-gain"]),
                bigrams=self.bigrams,
                lower_bounding=False,
                save_words=False,
                n_jobs=self.n_jobs,
            )

            # generate dilated dataset
            X2 = WEASEL_STEROIDS._dilation(X, dilation)

            # generate SFA words on subsample
            sfa_words = transformer.fit_transform(X2, y)

            feature_names = set()
            for t_words in sfa_words:
                for t_word in t_words:
                    feature_names.add(t_word)

            feature_count = min(
                self.max_feature_count // self.ensemble_size, len(feature_names)
            )
            relevant_features_idx = rng.choice(
                len(feature_names), replace=False, size=feature_count
            )
            relevant_features = dict(
                zip(
                    np.array(list(feature_names))[relevant_features_idx],
                    np.arange(feature_count),
                )
            )

            # merging bag-of-patterns of different window_sizes
            # to single bag-of-patterns with prefix indicating
            # the used window-length
            all_win_words = np.zeros((self.n_instances, feature_count), dtype=np.int32)
            for j in range(len(sfa_words)):
                for key in sfa_words[j]:
                    if key in relevant_features:
                        all_win_words[j, relevant_features[key]] += 1

            return (
                all_win_words,
                transformer,
                relevant_features,
                feature_count,
                dilation,
            )

        parallel_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit)(i) for i in range(self.ensemble_size)
        )

        features_count = 0
        all_words = np.zeros((len(X), self.max_feature_count), dtype=np.int32)

        for (
            sfa_words,
            transformer,
            relevant_features,
            rel_features_count,
            dilation,
        ) in parallel_res:
            self.SFA_transformers.append(transformer)
            self.dilation_factors.append(dilation)
            self.relevant_features.append(relevant_features)
            self.rel_features_counts.append(rel_features_count)

            # merging arrays from different threads
            for idx, bag in enumerate(sfa_words):
                all_words[
                    idx, features_count : (features_count + rel_features_count)
                ] = bag

            # print (features_count,":",(features_count+rel_features_count))
            features_count += rel_features_count

        # print("\tSize of dict", features_count)

        self.clf = make_pipeline(
            # DictVectorizer(sparse=True, sort=False),
            # GradientBoostingClassifier(
            #    subsample=0.8,
            #    min_samples_split=len(np.unique(y)),
            #    max_features="auto",
            #    random_state=self.random_state,
            #    n_estimators=200,
            #    init=LogisticRegression(
            #        solver="liblinear",
            #        penalty="l2",
            #        max_iter=5000,
            #        random_state=self.random_state
            #    )
            #    #init=KNeighborsClassifier(
            #    #    n_neighbors=5,
            #    #    algorithm='ball_tree')
            # )
            LogisticRegression(
                solver="liblinear",
                penalty="l2",
                max_iter=5000,
                random_state=self.random_state,
            )
        )

        self.clf.fit(all_words, y)
        # print("Train acc", self.clf.score(all_words, y))

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
        def _parallel_transform_words(
            X, transformer, relevant_features, feature_count, dilation
        ):

            X2 = WEASEL_STEROIDS._dilation(X, dilation)

            # SFA transform
            sfa_words = transformer.transform(X2)

            # merging arrays
            all_win_words = np.zeros((len(X), feature_count), dtype=np.int32)
            for j in range(len(sfa_words)):
                for key in sfa_words[j]:
                    if key in relevant_features:
                        all_win_words[j, relevant_features[key]] += 1

            return (all_win_words, feature_count)

        parallel_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_transform_words)(
                X,
                self.SFA_transformers[i],
                self.relevant_features[i],
                self.rel_features_counts[i],
                self.dilation_factors[i],
            )
            for i in range(self.ensemble_size)
        )

        """
        parallel_res = []
        for i in range(len(self.SFA_transformers)):
            parallel_res.append(
                self._parallel_transform_words(
                X,
                self.SFA_transformers[i],
                self.relevant_features[i],
                self.rel_features_counts[i],
                self.dilation_factors[i]))
        """

        features_count = 0
        all_words = np.zeros((len(X), self.max_feature_count), dtype=np.int32)
        for sfa_words, rel_features_count in parallel_res:
            for idx, bag in enumerate(sfa_words):
                all_words[
                    idx, features_count : (features_count + rel_features_count)
                ] = bag
            features_count += rel_features_count

        return all_words

    def _compute_window_inc(self):
        win_inc = self.window_inc
        if self.series_length < 100:
            win_inc = 1  # less than 100 is ok runtime-wise
        return win_inc

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
        return {"window_inc": 4}
