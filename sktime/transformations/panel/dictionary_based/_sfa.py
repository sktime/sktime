# -*- coding: utf-8 -*-
"""Symbolic Fourier Approximation (SFA) Transformer.

Configurable SFA transform for discretising time series into words.
"""

__author__ = ["MatthewMiddlehurst", "patrickzib"]
__all__ = ["SFA"]

import math
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import NumbaTypeSafetyWarning, njit, types
from numba.typed import Dict
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.panel import check_X

# The binning methods to use: equi-depth, equi-width, information gain or kmeans
binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans"}

# TODO remove imag-part from dc-component component


class SFA(BaseTransformer):
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

    binning_method:      {"equi-depth", "equi-width", "information-gain", "kmeans"},
                            default="equi-depth"
        the binning method used to derive the breakpoints.

    anova:               boolean, default = False
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given

    bigrams:             boolean, default = False
        whether to create bigrams of SFA words

    skip_grams:          boolean, default = False
        whether to create skip-grams of SFA words

    remove_repeat_words: boolean, default = False
        whether to use numerosity reduction (default False)

    levels:              int, default = 1
        Number of spatial pyramid levels

    save_words:          boolean, default = False
        whether to save the words generated for each series (default False)

    return_pandas_data_series:          boolean, default = False
        set to true to return Pandas Series as a result of transform.
        setting to true reduces speed significantly but is required for
        automatic test.

    n_jobs:              int, optional, default = 1
        The number of jobs to run in parallel for both `transform`.
        ``-1`` means using all processors.

    Attributes
    ----------
    words: []
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
    }

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        norm=False,
        binning_method="equi-depth",
        anova=False,
        bigrams=False,
        skip_grams=False,
        remove_repeat_words=False,
        levels=1,
        lower_bounding=True,
        save_words=False,
        keep_binning_dft=False,
        return_pandas_data_series=False,
        use_fallback_dft=False,
        typed_dict=False,
        n_jobs=1,
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        offset = 2 if norm else 0
        self.dft_length = window_size - offset if anova is True else word_length
        # make dft_length an even number (same number of reals and imags)
        self.dft_length = self.dft_length + self.dft_length % 2

        self.support = np.array(list(range(word_length)))

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
        self.keep_binning_dft = keep_binning_dft
        self.binning_dft = None

        self.levels = levels
        self.binning_method = binning_method
        self.anova = anova

        self.bigrams = bigrams
        self.skip_grams = skip_grams

        self.return_pandas_data_series = return_pandas_data_series
        self.use_fallback_dft = use_fallback_dft
        self._use_fallback_dft = (
            use_fallback_dft if word_length < window_size - offset else True
        )
        self.typed_dict = typed_dict

        self.n_jobs = n_jobs

        self.n_instances = 0
        self.series_length = 0
        self.letter_bits = 0
        self.letter_max = 0
        self.word_bits = 0
        self.max_bits = 0
        self.level_bits = 0
        self.level_max = 0

        super(SFA, self).__init__()

        if not return_pandas_data_series:
            self._output_convert = "off"

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
        if self.alphabet_size < 2:
            raise ValueError("Alphabet size must be an integer greater than 2")

        if self.word_length < 1:
            raise ValueError("Word length must be an integer greater than 1")

        if self.binning_method == "information-gain" and y is None:
            raise ValueError(
                "Class values must be provided for information gain binning"
            )

        if self.binning_method not in binning_methods:
            raise TypeError("binning_method must be one of: ", binning_methods)

        self.letter_bits = math.ceil(math.log2(self.alphabet_size))
        self.letter_max = pow(2, self.letter_bits) - 1
        self.word_bits = self.word_length * self.letter_bits
        self.max_bits = (
            self.word_bits * 2 if self.bigrams or self.skip_grams else self.word_bits
        )

        if self.typed_dict and self.max_bits > 64:
            raise ValueError(
                "Typed Dictionaries can only handle 64 bit words. "
                "ceil(log2(alphabet_size)) * word_length must be less than or equal "
                "to 64."
                "With bi-grams or skip-grams enabled, this bit limit is 32."
            )

        if self.typed_dict and self.levels > 15:
            raise ValueError(
                "Typed Dictionaries can only handle 15 levels "
                "(this is way to many anyway)."
            )

        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        if self.levels > 1:
            quadrants = 0
            for i in range(self.levels):
                quadrants += pow(2, i)
            self.level_bits = math.ceil(math.log2(quadrants))
            self.level_max = pow(2, self.level_bits) - 1

        self.n_instances, self.series_length = X.shape
        self.breakpoints = self._binning(X, y)

        self._is_fitted = True
        return self

    def _transform(self, X, y=None):
        """Transform data into SFA words.

        Parameters
        ----------
        X : pandas DataFrame or 3d numpy array, input time series.
        y : array_like, target values (optional, ignored).

        Returns
        -------
        List of dictionaries containing SFA words
        """
        X = X.squeeze(1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)
            transform = Parallel(n_jobs=self.n_jobs)(
                delayed(self._transform_case)(
                    X[i, :],
                    supplied_dft=self.binning_dft[i] if self.keep_binning_dft else None,
                )
                for i in range(X.shape[0])
            )

        dim, words = zip(*transform)
        if self.save_words:
            self.words = list(words)

        # cant pickle typed dict
        if self.typed_dict and self.n_jobs != 1:
            nl = [None] * len(dim)
            for i, pdict in enumerate(dim):
                ndict = (
                    Dict.empty(
                        key_type=types.UniTuple(types.int64, 2), value_type=types.uint32
                    )
                    if self.levels > 1
                    else Dict.empty(key_type=types.int64, value_type=types.uint32)
                )
                for key, val in pdict.items():
                    ndict[key] = val
                nl[i] = pdict
            dim = nl

        bags = pd.DataFrame() if self.return_pandas_data_series else [None]
        bags[0] = list(dim)

        return bags

    def _transform_case(self, X, supplied_dft=None):
        if supplied_dft is None:
            dfts = self._mft(X)
        else:
            dfts = supplied_dft

        if self.typed_dict:
            bag = (
                Dict.empty(
                    key_type=types.UniTuple(types.int64, 2), value_type=types.uint32
                )
                if self.levels > 1
                else Dict.empty(key_type=types.int64, value_type=types.uint32)
            )
        else:
            bag = {}

        last_word = -1
        repeat_words = 0
        words = (
            np.zeros(dfts.shape[0], dtype=np.int64)
            if self.word_bits <= 64
            else [0 for _ in range(dfts.shape[0])]
        )

        for window in range(dfts.shape[0]):
            word_raw = (
                SFA._create_word(
                    dfts[window],
                    self.word_length,
                    self.alphabet_size,
                    self.breakpoints,
                    self.letter_bits,
                )
                if self.word_bits <= 64
                else self._create_word_large(
                    dfts[window],
                )
            )
            words[window] = word_raw

            repeat_word = (
                self._add_to_pyramid(
                    bag, word_raw, last_word, window - int(repeat_words / 2)
                )
                if self.levels > 1
                else self._add_to_bag(bag, word_raw, last_word)
            )

            if repeat_word:
                repeat_words += 1
            else:
                last_word = word_raw
                repeat_words = 0

            if self.bigrams:
                if window - self.window_size >= 0:
                    bigram = self._create_bigram_words(
                        word_raw, words[window - self.window_size]
                    )

                    if self.levels > 1:
                        if self.typed_dict:
                            bigram = (bigram, -1)
                        else:
                            bigram = (bigram << self.level_bits) | 0
                    bag[bigram] = bag.get(bigram, 0) + 1

            if self.skip_grams:
                # creates skip-grams, skipping every (s-1)-th word in-between
                for s in range(2, 4):
                    if window - s * self.window_size >= 0:
                        skip_gram = self._create_bigram_words(
                            word_raw, words[window - s * self.window_size]
                        )

                        if self.levels > 1:
                            if self.typed_dict:
                                skip_gram = (skip_gram, -1)
                            else:
                                skip_gram = (skip_gram << self.level_bits) | 0
                        bag[skip_gram] = bag.get(skip_gram, 0) + 1

        # cant pickle typed dict
        if self.typed_dict and self.n_jobs != 1:
            pdict = dict()
            for key, val in bag.items():
                pdict[key] = val
            bag = pdict

        return [
            pd.Series(bag) if self.return_pandas_data_series else bag,
            words if self.save_words else [],
        ]

    def _binning(self, X, y=None):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        dft = np.array(
            [
                self._binning_dft(X[i, :], num_windows_per_inst)
                for i in range(self.n_instances)
            ]
        )
        if self.keep_binning_dft:
            self.binning_dft = dft
        dft = dft.reshape(len(X) * num_windows_per_inst, self.dft_length)

        if y is not None:
            y = np.repeat(y, num_windows_per_inst)

        if self.anova and y is not None:
            non_constant = np.where(
                ~np.isclose(dft.var(axis=0), np.zeros_like(dft.shape[1]))
            )[0]

            # select word-length many indices with best f-score
            if self.word_length <= non_constant.size:
                f, _ = f_classif(dft[:, non_constant], y)
                self.support = non_constant[np.argsort(-f)][: self.word_length]

            # sort remaining indices
            # self.support = np.sort(self.support)

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support) + 1
            self.dft_length = self.dft_length + self.dft_length % 2  # even

        if self.binning_method == "information-gain":
            return self._igb(dft, y)
        elif self.binning_method == "kmeans":
            return self._k_bins_discretizer(dft)
        else:
            return self._mcb(dft)

    def _k_bins_discretizer(self, dft):
        encoder = KBinsDiscretizer(
            n_bins=self.alphabet_size, strategy=self.binning_method
        )
        encoder.fit(dft)
        breaks = encoder.bin_edges_
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):
            for bp in range(1, len(breaks[letter]) - 1):
                breakpoints[letter][bp - 1] = breaks[letter][bp]

        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        return breakpoints

    def _mcb(self, dft):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        total_num_windows = int(self.n_instances * num_windows_per_inst)
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):
            res = [round(dft[i][letter] * 100) / 100 for i in range(total_num_windows)]
            column = np.sort(np.array(res))

            bin_index = 0

            # use equi-depth binning
            if self.binning_method == "equi-depth":
                target_bin_depth = total_num_windows / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    bin_index += target_bin_depth
                    breakpoints[letter][bp] = column[int(bin_index)]

            # use equi-width binning aka equi-frequency binning
            elif self.binning_method == "equi-width":
                target_bin_width = (column[-1] - column[0]) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    breakpoints[letter][bp] = (bp + 1) * target_bin_width + column[0]

        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        return breakpoints

    def _igb(self, dft, y):
        breakpoints = np.zeros((self.word_length, self.alphabet_size))
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=int(np.floor(np.log2(self.alphabet_size))),
            max_leaf_nodes=self.alphabet_size,
            random_state=1,
        )

        for i in range(self.word_length):
            clf.fit(dft[:, i][:, None], y)
            threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
            for bp in range(len(threshold)):
                breakpoints[i][bp] = threshold[bp]
            for bp in range(len(threshold), self.alphabet_size):
                breakpoints[i][bp] = sys.float_info.max

        return np.sort(breakpoints, axis=1)

    def _binning_dft(self, series, num_windows_per_inst):
        # Splits individual time series into windows and returns the DFT for
        # each
        split = np.split(
            series,
            np.linspace(
                self.window_size,
                self.window_size * (num_windows_per_inst - 1),
                num_windows_per_inst - 1,
                dtype=np.int_,
            ),
        )
        start = self.series_length - self.window_size
        split[-1] = series[start : self.series_length]

        result = np.zeros((len(split), self.dft_length), dtype=np.float64)

        for i, row in enumerate(split):
            result[i] = (
                self._discrete_fourier_transform(
                    row,
                    self.dft_length,
                    self.norm,
                    self.inverse_sqrt_win_size,
                    self.lower_bounding,
                )
                if self._use_fallback_dft
                else self._fast_fourier_transform(row)
            )

        return result

    def _fast_fourier_transform(self, series):
        """Perform a discrete fourier transform using the fast fourier transform.

        if self.norm is True, then the first term of the DFT is ignored

        Input
        -------
        X : The training input samples.  array-like or sparse matrix of
        shape = [n_samps, num_atts]

        Returns
        -------
        1D array of fourier term, real_0,imag_0, real_1, imag_1 etc, length
        num_atts or
        num_atts-2 if if self.norm is True
        """
        # first two are real and imaginary parts
        start = 2 if self.norm else 0

        s = np.std(series)
        std = s if s > 1e-8 else 1

        X_fft = np.fft.rfft(series)
        reals = np.real(X_fft)
        imags = np.imag(X_fft)

        length = start + self.dft_length
        dft = np.empty((length,), dtype=reals.dtype)
        dft[0::2] = reals[: np.uint32(length / 2)]
        dft[1::2] = imags[: np.uint32(length / 2)]
        if self.lower_bounding:
            dft[1::2] = dft[1::2] * -1  # lower bounding
        dft *= self.inverse_sqrt_win_size / std
        return dft[start:]

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _discrete_fourier_transform(
        series,
        dft_length,
        norm,
        inverse_sqrt_win_size,
        lower_bounding,
        apply_normalising_factor=True,
        cut_start_if_norm=True,
    ):
        """Perform a discrete fourier transform using standard O(n^2) transform.

        if self.norm is True, then the first term of the DFT is ignored

        Input
        -------
        X : The training input samples.  array-like or sparse matrix of
        shape = [n_samps, num_atts]

        Returns
        -------
        1D array of fourier term, real_0,imag_0, real_1, imag_1 etc, length
        num_atts or
        num_atts-2 if if self.norm is True
        """
        start = 2 if norm else 0
        output_length = start + dft_length

        if cut_start_if_norm:
            c = int(start / 2)
        else:
            c = 0
            start = 0

        dft = np.zeros(output_length - start)
        for i in range(c, int(output_length / 2)):
            for n in range(len(series)):
                dft[(i - c) * 2] += series[n] * math.cos(
                    2 * math.pi * n * i / len(series)
                )
                dft[(i - c) * 2 + 1] += -series[n] * math.sin(
                    2 * math.pi * n * i / len(series)
                )

        if apply_normalising_factor:
            if lower_bounding:
                dft[1::2] = dft[1::2] * -1  # lower bounding

            std = np.std(series)
            if std == 0:
                std = 1
            dft *= inverse_sqrt_win_size / std

        return dft

    def _mft(self, series):
        start_offset = 2 if self.norm else 0
        length = self.dft_length + start_offset + self.dft_length % 2
        end = max(1, len(series) - self.window_size + 1)

        phis = SFA._get_phis(self.window_size, length)
        stds = SFA._calc_incremental_mean_std(series, end, self.window_size)
        transformed = np.zeros((end, length))

        # first run with dft
        if self._use_fallback_dft:
            mft_data = self._discrete_fourier_transform(
                series[0 : self.window_size],
                self.dft_length,
                self.norm,
                self.inverse_sqrt_win_size,
                self.lower_bounding,
                apply_normalising_factor=False,
                cut_start_if_norm=False,
            )
        else:
            X_fft = np.fft.rfft(series[: self.window_size])
            reals = np.real(X_fft)
            imags = np.imag(X_fft)
            mft_data = np.empty((length,), dtype=reals.dtype)
            mft_data[0::2] = reals[: np.uint32(length / 2)]
            mft_data[1::2] = imags[: np.uint32(length / 2)]

        transformed[0] = mft_data * self.inverse_sqrt_win_size / stds[0]

        # other runs using mft
        # moved to external method to use njit
        SFA._iterate_mft(
            series,
            mft_data,
            phis,
            self.window_size,
            stds,
            transformed,
            self.inverse_sqrt_win_size,
        )

        if self.lower_bounding:
            transformed[:, 1::2] = transformed[:, 1::2] * -1  # lower bounding

        return (
            transformed[:, start_offset:][:, self.support]
            if self.anova
            else transformed[:, start_offset:]
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _get_phis(window_size, length):
        phis = np.zeros(length)
        for i in range(int(length / 2)):
            phis[i * 2] += math.cos(2 * math.pi * (-i) / window_size)
            phis[i * 2 + 1] += -math.sin(2 * math.pi * (-i) / window_size)
        return phis

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _iterate_mft(
        series, mft_data, phis, window_size, stds, transformed, inverse_sqrt_win_size
    ):
        for i in range(1, len(transformed)):
            for n in range(0, len(mft_data), 2):
                # only compute needed indices
                real = mft_data[n] + series[i + window_size - 1] - series[i - 1]
                imag = mft_data[n + 1]
                mft_data[n] = real * phis[n] - imag * phis[n + 1]
                mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag

            normalising_factor = inverse_sqrt_win_size / stds[i]
            transformed[i] = mft_data * normalising_factor

    def _shorten_bags(self, word_len):
        if self.save_words is False:
            raise ValueError(
                "Words from transform must be saved using save_word to shorten bags."
            )

        if word_len > self.word_length:
            word_len = self.word_length

        if self.typed_dict:
            warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

        dim = Parallel(n_jobs=self.n_jobs)(
            delayed(self._shorten_case)(word_len, i) for i in range(len(self.words))
        )

        # cant pickle typed dict
        if self.typed_dict and self.n_jobs != 1:
            nl = [None] * len(dim)
            for i, pdict in enumerate(dim):
                ndict = (
                    Dict.empty(
                        key_type=types.UniTuple(types.int64, 2), value_type=types.uint32
                    )
                    if self.levels > 1
                    else Dict.empty(key_type=types.int64, value_type=types.uint32)
                )
                for key, val in pdict.items():
                    ndict[key] = val
                nl[i] = pdict
            dim = nl

        new_bags = pd.DataFrame() if self.return_pandas_data_series else [None]
        new_bags[0] = list(dim)

        return new_bags

    def _shorten_case(self, word_len, i):
        if self.typed_dict:
            new_bag = (
                Dict.empty(
                    key_type=types.Tuple((types.int64, types.int16)),
                    value_type=types.uint32,
                )
                if self.levels > 1
                else Dict.empty(key_type=types.int64, value_type=types.uint32)
            )
        else:
            new_bag = {}

        last_word = -1
        repeat_words = 0

        for window, word in enumerate(self.words[i]):
            new_word = self._shorten_words(word, word_len)

            repeat_word = (
                self._add_to_pyramid(
                    new_bag, new_word, last_word, window - int(repeat_words / 2)
                )
                if self.levels > 1
                else self._add_to_bag(new_bag, new_word, last_word)
            )

            if repeat_word:
                repeat_words += 1
            else:
                last_word = new_word
                repeat_words = 0

            if self.bigrams:
                if window - self.window_size >= 0:
                    bigram = self._create_bigram_words(
                        new_word,
                        self._shorten_words(
                            self.words[i][window - self.window_size], word_len
                        ),
                    )

                    if self.levels > 1:
                        if self.typed_dict:
                            bigram = (bigram, -1)
                        else:
                            bigram = (bigram << self.level_bits) | 0
                    new_bag[bigram] = new_bag.get(bigram, 0) + 1

            if self.skip_grams:
                # creates skip-grams, skipping every (s-1)-th word in-between
                for s in range(2, 4):
                    if window - s * self.window_size >= 0:
                        skip_gram = self._create_bigram_words(
                            new_word,
                            self._shorten_words(
                                self.words[i][window - s * self.window_size],
                                word_len,
                            ),
                        )

                        if self.levels > 1:
                            if self.typed_dict:
                                skip_gram = (skip_gram, -1)
                            else:
                                skip_gram = (skip_gram << self.level_bits) | 0
                        new_bag[skip_gram] = new_bag.get(skip_gram, 0) + 1

        # cant pickle typed dict
        if self.typed_dict and self.n_jobs != 1:
            pdict = dict()
            for key, val in new_bag.items():
                pdict[key] = val
            new_bag = pdict

        return pd.Series(new_bag) if self.return_pandas_data_series else new_bag

    def _add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word == last_word:
            return True

        # store the histogram of word counts
        bag[word] = bag.get(word, 0) + 1

        return False

    def _add_to_pyramid(self, bag, word, last_word, window_ind):
        if self.remove_repeat_words and word == last_word:
            return True

        start = 0
        for i in range(self.levels):
            if self.typed_dict:
                new_word, num_quadrants = SFA._add_level_typed(
                    word, start, i, window_ind, self.window_size, self.series_length
                )
            else:
                new_word, num_quadrants = (
                    SFA._add_level(
                        word,
                        start,
                        i,
                        window_ind,
                        self.window_size,
                        self.series_length,
                        self.level_bits,
                    )
                    if self.word_bits + self.level_bits <= 64
                    else self._add_level_large(
                        word,
                        start,
                        i,
                        window_ind,
                    )
                )
            bag[new_word] = bag.get(new_word, 0) + num_quadrants
            start += num_quadrants

        return False

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _add_level(
        word, start, level, window_ind, window_size, series_length, level_bits
    ):
        num_quadrants = pow(2, level)
        quadrant = start + int(
            (window_ind + int(window_size / 2)) / int(series_length / num_quadrants)
        )
        return (word << level_bits) | quadrant, num_quadrants

    def _add_level_large(self, word, start, level, window_ind):
        num_quadrants = pow(2, level)
        quadrant = start + int(
            (window_ind + int(self.window_size / 2))
            / int(self.series_length / num_quadrants)
        )
        return (word << self.level_bits) | quadrant, num_quadrants

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _add_level_typed(word, start, level, window_ind, window_size, series_length):
        num_quadrants = pow(2, level)
        quadrant = start + int(
            (window_ind + int(window_size / 2)) / int(series_length / num_quadrants)
        )
        return (word, quadrant), num_quadrants

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _create_word(dft, word_length, alphabet_size, breakpoints, letter_bits):
        word = np.int64(0)
        for i in range(word_length):
            for bp in range(alphabet_size):
                if dft[i] <= breakpoints[i][bp]:
                    word = (word << letter_bits) | bp
                    break

        return word

    def _create_word_large(self, dft):
        word = 0
        for i in range(self.word_length):
            for bp in range(self.alphabet_size):
                if dft[i] <= self.breakpoints[i][bp]:
                    word = (word << self.letter_bits) | bp
                    break

        return word

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _calc_incremental_mean_std(series, end, window_size):
        stds = np.zeros(end)
        window = series[0:window_size]
        series_sum = np.sum(window)
        square_sum = np.sum(np.multiply(window, window))

        r_window_length = 1 / window_size
        mean = series_sum * r_window_length
        buf = math.sqrt(square_sum * r_window_length - mean * mean)
        stds[0] = buf if buf > 1e-8 else 1

        for w in range(1, end):
            series_sum += series[w + window_size - 1] - series[w - 1]
            mean = series_sum * r_window_length
            square_sum += (
                series[w + window_size - 1] * series[w + window_size - 1]
                - series[w - 1] * series[w - 1]
            )
            buf = math.sqrt(square_sum * r_window_length - mean * mean)
            stds[w] = buf if buf > 1e-8 else 1

        return stds

    def _create_bigram_words(self, word_raw, word):
        return (
            SFA._create_bigram_word(
                word,
                word_raw,
                self.word_bits,
            )
            if self.max_bits <= 64
            else self._create_bigram_word_large(
                word,
                word_raw,
            )
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _create_bigram_word(word, other_word, word_bits):
        return (word << word_bits) | other_word

    def _create_bigram_word_large(self, word, other_word):
        return (word << self.word_bits) | other_word

    def _shorten_words(self, word, word_len):
        return (
            SFA._shorten_word(word, self.word_length - word_len, self.letter_bits)
            if self.word_bits <= 64
            else self._shorten_word_large(word, self.word_length - word_len)
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _shorten_word(word, amount, letter_bits):
        # shorten a word by set amount of letters
        return word >> amount * letter_bits

    def _shorten_word_large(self, word, amount):
        # shorten a word by set amount of letters
        return word >> amount * self.letter_bits

    def bag_to_string(self, bag):
        """Convert a bag of SFA words into a string."""
        s = "{"
        for word, value in bag.items():
            s += "{0}: {1}, ".format(
                self.word_list_typed(word) if self.typed_dict else self.word_list(word),
                value,
            )
        s = s[:-2]
        return s + "}"

    def word_list(self, word):
        """Find list of integers to obtain input word."""
        letters = []
        word_bits = self.word_bits + self.level_bits
        shift = word_bits - self.letter_bits

        for _ in range(self.word_length, 0, -1):
            letters.append(word >> shift & self.letter_max)
            shift -= self.letter_bits

        if word.bit_length() > self.word_bits + self.level_bits:
            bigram_letters = []
            shift = self.word_bits + word_bits - self.letter_bits
            for _ in range(self.word_length, 0, -1):
                bigram_letters.append(word >> shift & self.letter_max)
                shift -= self.letter_bits

            letters = (bigram_letters, letters)

        if self.levels > 1:
            level = word >> 0 & self.level_max
            letters = (letters, level)

        return letters

    def word_list_typed(self, word):
        """Find list of integers to obtain input word."""
        letters = []
        shift = self.word_bits - self.letter_bits

        if self.levels > 1:
            level = word[1]
            word = word[0]

        for _ in range(self.word_length, 0, -1):
            letters.append(word >> shift & self.letter_max)
            shift -= self.letter_bits

        if word.bit_length() > self.word_bits:
            bigram_letters = []
            shift = self.word_bits + self.word_bits - self.letter_bits
            for _ in range(self.word_length, 0, -1):
                bigram_letters.append(word >> shift & self.letter_max)
                shift -= self.letter_bits

            letters = (bigram_letters, letters)

        if self.levels > 1:
            letters = (letters, level)

        return letters

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
        params = {"window_size": 4, "return_pandas_data_series": True}
        return params
