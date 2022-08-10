# -*- coding: utf-8 -*-
"""WEASEL on Steroids classifier.

Dictionary based classifier based on SFA transform, BOSS and linear regression.

The fastest WEASEL on earth
Tour de WEASEL
WEASEL on steroids

"""

__author__ = ["patrickzib"]
__all__ = ["WEASEL_STEROIDS"]

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV

# from numba import njit
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA_NEW


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
        "capability:multithreading": True,
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
        self.dilation_factors = []
        self.relevant_features = []
        self.rel_features_counts = []
        self.first_differences = []
        self.clf = None
        self.n_jobs = n_jobs

        super(WEASEL_STEROIDS, self).__init__()

    @staticmethod
    # @njit
    def _dilation(X, d, ws, first_difference):
        # padding = np.zeros((len(X), d // 2))
        # X = np.concatenate((padding, X, padding), axis=1)

        if first_difference:
            X2 = np.diff(X, axis=1)
            X = np.concatenate((X, X2), axis=1)

        # dilation on actual data
        X_first = np.array(X)[:, 0::d]
        for i in range(1, d):
            X_second = X[:, i::d]
            X_first = np.concatenate((X_first, X_second), axis=1)

        # if not first_difference:
        return X_first

        """
        # dilation on first order differences
        if first_difference:
            X2 = np.diff(X, axis=1)
            X2_first = X2[:, 0::d]
            for i in range(1, d):
                X2_second = X2[:, i::d]
                X2_first = np.concatenate((X2_first, X2_second), axis=1)
            return np.concatenate((X_first, X2_first), axis=1)
        """

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
        self.window_sizes = list(range(self.min_window, self.max_window + 1, 1))

        def _parallel_fit(
            i,
        ):
            rng = check_random_state(i)
            window_size = rng.choice(self.window_sizes)
            alphabet_size = rng.choice(self.alphabet_sizes)
            word_length = min(window_size - 2, rng.choice(self.word_lengths))
            norm = rng.choice(self.norm_options)
            dilation = np.int32(
                2
                ** rng.uniform(0, np.log2((self.series_length - 1) / (window_size - 1)))
            )
            first_difference = rng.choice(self.use_first_differences)
            binning_strategy = rng.choice(self.binning_strategies)

            # TODO count subgroups of two letters of the words?
            # TODO test different configurations?

            transformer = SFA_NEW(
                variance=self.variance,
                word_length=word_length,
                alphabet_size=alphabet_size,
                window_size=window_size,
                norm=norm,
                anova=self.anova,
                binning_method=binning_strategy,
                bigrams=self.bigrams,
                lower_bounding=False,
                n_jobs=self.n_jobs,
            )

            # generate dilated dataset
            X2 = WEASEL_STEROIDS._dilation(X, dilation, window_size, first_difference)

            # generate SFA words on subsample
            sfa_words = transformer.fit_transform(X2, y)

            # all feature names
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

            # merging of arrays of used window-length
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
                first_difference,
            )

        parallel_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit)(i) for i in range(self.ensemble_size)
        )

        features_count = 0
        all_words = np.zeros((len(X), self.max_feature_count), dtype=np.float32)

        for (
            sfa_words2,
            transformer2,
            relevant_features2,
            rel_features_count2,
            dilation2,
            first_difference2,
        ) in parallel_res:
            self.SFA_transformers.append(transformer2)
            self.dilation_factors.append(dilation2)
            self.relevant_features.append(relevant_features2)
            self.rel_features_counts.append(rel_features_count2)
            self.first_differences.append(first_difference2)

            # merging arrays from different threads
            for idx, bag in enumerate(sfa_words2):
                all_words[
                    idx, features_count : (features_count + rel_features_count2)
                ] = bag

            features_count += rel_features_count2

        # all_words = all_words / np.max(all_words)

        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=False)

        # self.clf = make_pipeline(
        #    StandardScaler(with_mean=True),
        #    RidgeClassifierCV(
        #        alphas=np.logspace(-3, 3, 10),
        #        class_weight=None,
        #        fit_intercept=True
        #    )
        # )
        # make_pipeline(
        # GradientBoostingClassifier(
        #   subsample=0.8,
        #   min_samples_split=len(np.unique(y)),
        #   max_features="auto",
        #   random_state=self.random_state,
        #   n_estimators=200,
        #   init=LogisticRegression(
        #       solver="liblinear",
        #       penalty="l2",
        #       max_iter=5000,
        #       random_state=self.random_state
        #   )
        # )
        # LogisticRegression(
        #    solver="liblinear",
        #    penalty="l2",
        #    max_iter=5000,
        #    random_state=self.random_state,
        # )
        # )

        self.clf.fit(all_words, y)
        # print("Train acc", self.clf.score(all_words, y))
        # print(self.clf.cv_values_)

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
            X, transformer, relevant_features, feature_count, dilation, first_difference
        ):
            X2 = WEASEL_STEROIDS._dilation(
                X, dilation, transformer.window_size, first_difference
            )

            # SFA transform
            sfa_words = transformer.transform(X2)

            # merging arrays
            all_win_words = np.zeros((len(X), feature_count), dtype=np.int32)
            for j in range(len(sfa_words)):
                for key in sfa_words[j]:
                    if key in relevant_features:
                        all_win_words[j, relevant_features[key]] += 1

            return all_win_words, feature_count

        X = X.squeeze(1)

        parallel_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_transform_words)(
                X,
                self.SFA_transformers[i],
                self.relevant_features[i],
                self.rel_features_counts[i],
                self.dilation_factors[i],
                self.first_differences[i],
            )
            for i in range(self.ensemble_size)
        )

        features_count = 0
        all_words = np.zeros((len(X), self.max_feature_count), dtype=np.float32)
        for sfa_words, rel_features_count in parallel_res:
            for idx, bag in enumerate(sfa_words):
                all_words[
                    idx, features_count : (features_count + rel_features_count)
                ] = bag
            features_count += rel_features_count

        # all_words = all_words / np.max(all_words)
        return all_words

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
