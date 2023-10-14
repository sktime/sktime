"""Symbolic Fourier Approximation (SFA) Transformer.

Configurable SFA transform for discretising time series into words.
"""

__author__ = ["patrickzib"]
__all__ = ["SFAFast"]

import math
import multiprocessing
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.panel import check_X

# The binning methods to use: equi-depth, equi-width, information gain or kmeans
binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans", "quantile"}


class SFAFast(BaseTransformer):
    """Symbolic Fourier Approximation (SFA) Transformer.

    Overview: for each series:
        run a sliding window across the series
        for each window
            shorten the series with DFT
            discretise the shortened series into bins set by MFC
            form a word from these discrete values
    by default SFA produces a single word per series (window_size=0)
    if a window is used, it forms a histogram of counts of words.

    Parameters
    ----------
        word_length:         int, default = 8
            length of word to shorten window to (using PAA)

        alphabet_size:       int, default = 4
            number of values to discretise each value to

        window_size:         int, default = 12
            size of window for sliding. Input series
            length for whole series transform

        norm:                boolean, default = False
            mean normalise words by dropping first fourier coefficient

        binning_method:      {"equi-depth", "equi-width", "information-gain", "kmeans",
                              "quantile"},
                             default="equi-depth"
            the binning method used to derive the breakpoints.

        anova:               boolean, default = False
            If True, the Fourier coefficient selection is done via a one-way
            ANOVA test. If False, the first Fourier coefficients are selected.
            Only applicable if labels are given

        variance:               boolean, default = False
            If True, the Fourier coefficient selection is done via the largest
            variance. If False, the first Fourier coefficients are selected.
            Only applicable if labels are given

        save_words:          boolean, default = False
            whether to save the words generated for each series (default False)

        bigrams:             boolean, default = False
            whether to create bigrams of SFA words

        feature_selection: {"chi2", "none", "random"}, default: chi2
            Sets the feature selections strategy to be used. Chi2 reduces the number
            of words significantly and is thus much faster (preferred). Random also
            reduces the number significantly. None applies not feature selectiona and
            yields large bag of words, e.g. much memory may be needed.

        p_threshold:  int, default=0.05 (disabled by default)
            If feature_selection=chi2 is chosen, feature selection is applied based on
            the chi-squared test. This is the p-value threshold to use for chi-squared
            test on bag-of-words (lower means more strict). 1 indicates that the test
            should not be performed.

        max_feature_count:  int, default=256
            If feature_selection=random is chosen, this parameter defines the number of
            randomly chosen unique words used.

        skip_grams:     boolean, default = False
            whether to create skip-grams of SFA words

        remove_repeat_words: boolean, default = False
            whether to use numerosity reduction (default False)

        return_sparse:  boolean, default=True
            if set to true, a scipy sparse matrix will be returned as BOP model.
            If set to false a dense array will be returned as BOP model. Sparse
            arrays are much more compact.

        n_jobs:     int, optional, default = 1
            The number of jobs to run in parallel for both `transform`.
            ``-1`` means using all processors.

        return_pandas_data_series:          boolean, default = False
            set to true to return Pandas Series as a result of transform.
            setting to true reduces speed significantly but is required for
            automatic test.

    Attributes
    ----------
    breakpoints: = []
    num_insts = 0
    num_atts = 0


    References
    ----------
    .. [1] Schäfer, Patrick, and Mikael Högqvist. "SFA: a symbolic fourier approximation
    and  index for similarity search in high dimensional datasets." Proceedings of the
    15th international conference on extending database technology. 2012.
    """

    _tags = {
        "univariate-only": True,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd_Series_Table",  # which mtypes does y require?
        "requires_y": True,  # does y need to be passed in fit?
        "python_dependencies": ["numba", "scipy"],
    }

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        norm=False,
        binning_method="equi-depth",
        anova=False,
        variance=False,
        bigrams=False,
        skip_grams=False,
        remove_repeat_words=False,
        lower_bounding=True,
        save_words=False,
        feature_selection="none",
        max_feature_count=256,
        p_threshold=0.05,
        random_state=None,
        return_sparse=True,
        return_pandas_data_series=False,
        n_jobs=1,
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        self.word_length = word_length

        self.alphabet_size = alphabet_size
        self.window_size = window_size

        self.norm = norm
        self.lower_bounding = lower_bounding
        self.inverse_sqrt_win_size = (
            1.0 / math.sqrt(window_size) if not lower_bounding else 1.0
        )

        self.remove_repeat_words = remove_repeat_words

        self.save_words = save_words

        self.binning_method = binning_method
        self.anova = anova
        self.variance = variance

        self.bigrams = bigrams
        self.skip_grams = skip_grams
        self.n_jobs = n_jobs

        self.n_instances = 0
        self.series_length = 0
        self.letter_bits = 0

        # Feature selection part
        self.feature_selection = feature_selection
        self.max_feature_count = max_feature_count
        self.feature_count = 0
        self.relevant_features = None

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold

        self.return_sparse = return_sparse
        self.return_pandas_data_series = return_pandas_data_series

        self.random_state = random_state

        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        super().__init__()
        # super raises numba import exception if not available
        # so now we know we can use numba

        from numba import set_num_threads

        set_num_threads(n_jobs)

        if not return_pandas_data_series:
            self.set_config(**{"output_conversion": "off"})

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        from sktime.transformations.panel.dictionary_based._sfa_fast_numba import (
            _transform_case,
            remove_repeating_words,
        )

        if self.alphabet_size < 2:
            raise ValueError("Alphabet size must be an integer greater than 2")

        if self.binning_method == "information-gain" and y is None:
            raise ValueError(
                "Class values must be provided for information gain binning"
            )

        if self.variance and self.anova:
            raise ValueError(
                "Please set either variance or anova Fourier coefficient selection"
            )

        if self.binning_method not in binning_methods:
            raise TypeError("binning_method must be one of: ", binning_methods)

        offset = 2 if self.norm else 0
        self.word_length_actual = min(self.window_size - offset, self.word_length)
        self.dft_length = (
            self.window_size - offset
            if (self.anova or self.variance) is True
            else self.word_length_actual
        )
        # make dft_length an even number (same number of reals and imags)
        self.dft_length = self.dft_length + self.dft_length % 2
        self.word_length_actual = self.word_length_actual + self.word_length_actual % 2

        self.support = np.arange(self.word_length_actual)
        self.letter_bits = np.uint32(math.ceil(math.log2(self.alphabet_size)))
        # self.word_bits = self.word_length_actual * self.letter_bits

        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        self.n_instances, self.series_length = X.shape
        self.breakpoints = self._binning(X, y)
        self._is_fitted = True

        words, dfts = _transform_case(
            X,
            self.window_size,
            self.dft_length,
            self.word_length_actual,
            self.norm,
            self.remove_repeat_words,
            self.support,
            self.anova,
            self.variance,
            self.breakpoints,
            self.letter_bits,
            self.bigrams,
            self.skip_grams,
            self.inverse_sqrt_win_size,
            self.lower_bounding,
        )

        if self.remove_repeat_words:
            words = remove_repeating_words(words)

        if self.save_words:
            self.words = words

        # fitting: learns the feature selection strategy, too
        return self.transform_to_bag(words, self.word_length_actual, y)

    def fit(self, X, y=None):
        """Calculate word breakpoints using MCB or IGB.

        Parameters
        ----------
        X : pandas DataFrame or 3d numpy array, input time series.
        y : array_like, target values (optional, ignored).

        Returns
        -------
        self: object
        """
        # with parallel_backend("loky", inner_max_num_threads=n_jobs):
        self.fit_transform(X, y)
        return self

    def transform(self, X, y=None):
        """Transform data into SFA words.

        Parameters
        ----------
        X : pandas DataFrame or 3d numpy array, input time series.
        y : array_like, target values (optional, ignored).

        Returns
        -------
        List of dictionaries containing SFA words
        """
        from numba.core import types
        from numba.typed import Dict
        from scipy.sparse import csr_matrix

        from sktime.transformations.panel.dictionary_based._sfa_fast_numba import (
            _transform_case,
            create_bag_transform,
        )

        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        words, dfts = _transform_case(
            X,
            self.window_size,
            self.dft_length,
            self.word_length_actual,
            self.norm,
            self.remove_repeat_words,
            self.support,
            self.anova,
            self.variance,
            self.breakpoints,
            self.letter_bits,
            self.bigrams,
            self.skip_grams,
            self.inverse_sqrt_win_size,
            self.lower_bounding,
        )

        # only save at fit
        # if self.save_words:
        #    self.words = words

        # transform: applies the feature selection strategy
        empty_dict = Dict.empty(
            key_type=types.uint32,
            value_type=types.uint32,
        )

        # transform
        bags = create_bag_transform(
            self.feature_count,
            self.feature_selection,
            self.relevant_features if self.relevant_features else empty_dict,
            words,
            self.bigrams,
            self.remove_repeat_words,
        )[0]

        if self.return_pandas_data_series:
            bb = pd.DataFrame()
            bb[0] = [pd.Series(bag) for bag in bags]
            return bb
        elif self.return_sparse:
            bags = csr_matrix(bags, dtype=np.uint32)
        return bags

    def transform_to_bag(self, words, word_len, y=None):
        """Transform words to bag-of-pattern and apply feature selection."""
        from numba.core import types
        from numba.typed import Dict
        from scipy.sparse import csr_matrix

        from sktime.transformations.panel.dictionary_based._sfa_fast_numba import (
            create_bag_feature_selection,
            create_bag_none,
            create_feature_names,
        )

        bag_of_words = None
        rng = check_random_state(self.random_state)

        if self.feature_selection == "none" and (
            self.breakpoints.shape[1] <= 2 and not self.bigrams
        ):
            bag_of_words = create_bag_none(
                self.breakpoints,
                words.shape[0],
                words,
                word_len,  # self.word_length_actual,
                self.remove_repeat_words,
            )
        else:
            feature_names = create_feature_names(words)

            if self.feature_selection == "none":
                feature_count = len(list(feature_names))
                relevant_features_idx = np.arange(feature_count, dtype=np.uint32)
                bag_of_words, self.relevant_features = create_bag_feature_selection(
                    words.shape[0],
                    relevant_features_idx,
                    np.array(list(feature_names)),
                    words,
                    self.remove_repeat_words,
                )

            # Random feature selection
            elif self.feature_selection == "random":
                feature_count = min(self.max_feature_count, len(feature_names))
                relevant_features_idx = rng.choice(
                    len(feature_names), replace=False, size=feature_count
                )
                bag_of_words, self.relevant_features = create_bag_feature_selection(
                    words.shape[0],
                    relevant_features_idx,
                    np.array(list(feature_names)),
                    words,
                    self.remove_repeat_words,
                )

            # Chi-squared feature selection
            elif self.feature_selection == "chi2":
                feature_count = len(list(feature_names))
                relevant_features_idx = np.arange(feature_count, dtype=np.uint32)
                bag_of_words, _ = create_bag_feature_selection(
                    words.shape[0],
                    relevant_features_idx,
                    np.array(list(feature_names)),
                    words,
                    self.remove_repeat_words,
                )

                chi2_statistics, p = chi2(bag_of_words, y)
                relevant_features_idx = np.where(p <= self.p_threshold)[0]
                self.relevant_features = Dict.empty(
                    key_type=types.uint32, value_type=types.uint32
                )
                for k, v in zip(
                    np.array(list(feature_names))[relevant_features_idx],
                    np.arange(len(relevant_features_idx)),
                ):
                    self.relevant_features[k] = v

                # select subset of features
                bag_of_words = bag_of_words[:, relevant_features_idx]

        self.feature_count = bag_of_words.shape[1]

        if self.return_pandas_data_series:
            bb = pd.DataFrame()
            bb[0] = [pd.Series(bag) for bag in bag_of_words]
            return bb
        elif self.return_sparse:
            bag_of_words = csr_matrix(bag_of_words, dtype=np.uint32)
        return bag_of_words

    def _binning(self, X, y=None):
        from sktime.transformations.panel.dictionary_based._sfa_fast_numba import (
            _binning_dft,
        )

        dft = _binning_dft(
            X,
            self.window_size,
            self.series_length,
            self.dft_length,
            self.norm,
            self.inverse_sqrt_win_size,
            self.lower_bounding,
        )

        if y is not None:
            y = np.repeat(y, dft.shape[0] / len(y))

        if self.variance and y is not None:
            # determine variance
            dft_variance = np.var(dft, axis=0)

            # select word-length-many indices with the largest variance
            self.support = np.argsort(-dft_variance)[: self.word_length_actual]

            # sort remaining indices
            self.support = np.sort(self.support)

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support) + 1
            self.dft_length = self.dft_length + self.dft_length % 2  # even

        if self.anova and y is not None:
            non_constant = np.where(
                ~np.isclose(dft.var(axis=0), np.zeros_like(dft.shape[1]))
            )[0]

            # select word-length many indices with best f-score
            if self.word_length_actual <= non_constant.size:
                f, _ = f_classif(dft[:, non_constant], y)
                self.support = non_constant[np.argsort(-f)][: self.word_length_actual]

            # sort remaining indices
            self.support = np.sort(self.support)

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support) + 1
            self.dft_length = self.dft_length + self.dft_length % 2  # even

        if self.binning_method == "information-gain":
            return self._igb(dft, y)
        elif self.binning_method == "kmeans" or self.binning_method == "quantile":
            return self._k_bins_discretizer(dft)
        else:
            return self._mcb(dft)

    def _k_bins_discretizer(self, dft):
        encoder = KBinsDiscretizer(
            n_bins=self.alphabet_size, strategy=self.binning_method
        )
        encoder.fit(dft)
        if encoder.bin_edges_.ndim == 1:
            breaks = encoder.bin_edges_.reshape((-1, 1))
        else:
            breaks = encoder.bin_edges_
        breakpoints = np.zeros((self.word_length_actual, self.alphabet_size))

        for letter in range(self.word_length_actual):
            for bp in range(1, len(breaks[letter]) - 1):
                breakpoints[letter, bp - 1] = breaks[letter, bp]

        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        return breakpoints

    def _mcb(self, dft):
        breakpoints = np.zeros((self.word_length_actual, self.alphabet_size))

        dft = np.round(dft, 2)
        for letter in range(self.word_length_actual):
            column = np.sort(dft[:, letter])
            bin_index = 0

            # use equi-depth binning
            if self.binning_method == "equi-depth":
                target_bin_depth = len(dft) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    bin_index += target_bin_depth
                    breakpoints[letter, bp] = column[int(bin_index)]

            # use equi-width binning aka equi-frequency binning
            elif self.binning_method == "equi-width":
                target_bin_width = (column[-1] - column[0]) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    breakpoints[letter, bp] = (bp + 1) * target_bin_width + column[0]

        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        return breakpoints

    def _igb(self, dft, y):
        breakpoints = np.zeros((self.word_length_actual, self.alphabet_size))
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=np.uint32(np.log2(self.alphabet_size)),
            max_leaf_nodes=self.alphabet_size,
            random_state=1,
        )

        for i in range(self.word_length_actual):
            clf.fit(dft[:, i][:, None], y)
            threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
            for bp in range(len(threshold)):
                breakpoints[i, bp] = threshold[bp]
            for bp in range(len(threshold), self.alphabet_size):
                breakpoints[i, bp] = np.inf

        return np.sort(breakpoints, axis=1)

    def _shorten_bags(self, word_len, y):
        from sktime.transformations.panel.dictionary_based._sfa_fast_numba import (
            shorten_words,
        )

        if self.save_words is False:
            raise ValueError(
                "Words from transform must be saved using save_word to shorten bags."
            )
        if self.bigrams:
            raise ValueError("Bigrams are currently not supported.")
        if self.variance or self.anova:
            raise ValueError(
                "Variance or Anova based feature selection is currently not supported."
            )

        # determine the new word-length
        new_len = min(word_len, self.word_length_actual)

        # the difference in word-length to shorten the words to
        new_len_diff = self.word_length_actual - new_len

        if new_len_diff > 0:
            new_words = shorten_words(self.words, new_len_diff, self.letter_bits)
        else:
            new_words = self.words

        # retrain feature selection-strategy
        return self.transform_to_bag(new_words, new_len, y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # small window size for testing
        params = {
            "word_length": 4,
            "window_size": 4,
            "return_sparse": True,
            "return_pandas_data_series": True,
            "feature_selection": "chi2",
            "alphabet_size": 2,
        }
        return params

    def set_fitted(self):
        """Whether `fit` has been called."""
        self._is_fitted = True

    def __getstate__(self):
        """Return state as dictionary for pickling, required for typed Dict objects."""
        from numba.typed import Dict

        state = self.__dict__.copy()

        if isinstance(state["relevant_features"], Dict):
            state["relevant_features"] = dict(state["relevant_features"])
        return state

    def __setstate__(self, state):
        """Set current state using input pickling, required for typed Dict objects."""
        from numba.core import types
        from numba.typed import Dict

        self.__dict__.update(state)
        if isinstance(self.relevant_features, dict):
            typed_dict = Dict.empty(key_type=types.uint32, value_type=types.uint32)
            for key, value in self.relevant_features.items():
                typed_dict[key] = value
            self.relevant_features = typed_dict
