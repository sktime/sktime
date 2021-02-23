# -*- coding: utf-8 -*-
__author__ = ["Matthew Middlehurst", "Patrick Schäfer"]
__all__ = ["SFA"]

import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X
from sklearn.preprocessing import KBinsDiscretizer

# from numba import typeof
# from numba.core import types
# from numba.typed import Dict

# The binning methods to use: equi-depth, equi-width or information gain
binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans"}

# TODO remove imag-part from dc-component component


class SFA(_PanelToPanelTransformer):
    """SFA (Symbolic Fourier Approximation) Transformer, as described in

    @inproceedings{schafer2012sfa,
      title={SFA: a symbolic fourier approximation and index for similarity
      search in high dimensional datasets},
      author={Sch{\"a}fer, Patrick and H{\"o}gqvist, Mikael},
      booktitle={Proceedings of the 15th International Conference on
      Extending Database Technology},
      pages={516--527},
      year={2012},
      organization={ACM}
    }

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
    """

    _tags = {"univariate-only": True}

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
        save_binning_dft=False,
        return_pandas_data_series=False,
        n_jobs=1,
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        offset = 2 if norm else 0
        self.word_length = min(word_length, window_size - offset)
        self.dft_length = window_size - offset if anova is True else self.word_length

        # make dft_length an even number (same number of reals and imags)
        self.dft_length = self.dft_length + self.dft_length % 2

        self.support = np.array(list(range(self.word_length)))

        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.lower_bounding = lower_bounding
        self.inverse_sqrt_win_size = (
            1.0 / math.sqrt(window_size) if lower_bounding else 1.0
        )

        self.norm = norm
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words

        # TDE
        self.levels = levels
        self.save_binning_dft = save_binning_dft
        self.binning_dft = None

        #
        self.binning_method = binning_method
        self.anova = anova

        self.bigrams = bigrams
        self.skip_grams = skip_grams

        # weighting for levels going up to 10 levels
        # No real reason to go past 3, should probably not weight with too many.
        self.level_weights = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        self.n_instances = 0
        self.series_length = 0
        self.return_pandas_data_series = return_pandas_data_series

        self.n_jobs = n_jobs

        super(SFA, self).__init__()

    def fit(self, X, y=None):
        """Calculate word breakpoints using _mcb

        Parameters
        ----------
        X: nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The class labels.

        Returns
        -------
        self: object
        """

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise ValueError("Alphabet size must be an integer between 2 and 4")

        if self.word_length < 1 or self.word_length > 16:
            raise ValueError("Word length must be an integer between 1 and 16")

        if self.binning_method == "information-gain" and y is None:
            raise ValueError(
                "Class values must be provided for information gain binning"
            )

        if self.binning_method not in binning_methods:
            raise TypeError("binning_method must be one of: ", binning_methods)

        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        self.n_instances, self.series_length = X.shape
        self.breakpoints = self._binning(X, y)

        self._is_fitted = True
        return self

    def transform(self, X, y=None, supplied_dft=None):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        transform = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform_case)(
                X,
                i,
                supplied_dft,
            )
            for i in range(X.shape[0])
        )

        dim, words = zip(*transform)
        if self.save_words:
            self.words = list(words)
        bags = pd.DataFrame() if self.return_pandas_data_series else [None]
        bags[0] = list(dim)

        return bags

    def _transform_case(self, X, i, supplied_dft):
        if supplied_dft is None:
            dfts = self._mft(X[i, :])
        else:
            dfts = supplied_dft[i]

        bag = defaultdict(int)
        # bag = Dict.empty(key_type=typeof((100,100.0)),
        #                  value_type=types.float64) \
        #     if self.levels > 1 else \
        #     Dict.empty(key_type=types.int64, value_type=types.int64)

        last_word = -1
        repeat_words = 0
        words = np.zeros(dfts.shape[0], dtype=np.int64)

        for window in range(dfts.shape[0]):
            word_raw = SFA._create_word(
                dfts[window], self.word_length, self.alphabet_size, self.breakpoints
            )
            words[window] = word_raw

            repeat_word = (
                self._add_to_pyramid(
                    bag, word_raw, last_word, window - int(repeat_words / 2)
                )
                if self.levels > 1
                else self._add_to_bag(bag, word_raw, last_word, window)
            )

            if repeat_word:
                repeat_words += 1
            else:
                last_word = word_raw
                repeat_words = 0

            if self.bigrams:
                if window - self.window_size >= 0:
                    bigram = self.create_bigram_word(
                        words[window - self.window_size],
                        word_raw,
                        self.word_length,
                    )

                    if self.levels > 1:
                        bigram = (bigram, 0)
                    bag[bigram] += 1

            if self.skip_grams:
                # creates skip-grams, skipping every (s-1)-th word in-between
                for s in range(2, 4):
                    if window - s * self.window_size >= 0:
                        skip_gram = self.create_bigram_word(
                            words[window - s * self.window_size],
                            word_raw,
                            self.word_length,
                        )

                        if self.levels > 1:
                            skip_gram = (skip_gram, 0)
                        bag[skip_gram] += 1

        return [
            pd.Series(bag) if self.return_pandas_data_series else bag,
            words if self.save_words else [],
        ]

    def _binning(self, X, y=None):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        dft = np.array(
            [
                self._mcb_dft(X[i, :], num_windows_per_inst)
                for i in range(self.n_instances)
            ]
        )
        if self.save_binning_dft:
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
            return self._KBinsDiscretizer(dft)
        else:
            return self._mcb(dft)

    def _KBinsDiscretizer(self, dft):
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
        total_num_windows = self.n_instances * num_windows_per_inst
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):

            res = [
                round(dft[inst][letter] * 100) / 100
                for inst in range(self.n_instances * num_windows_per_inst)
            ]
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

            breakpoints[letter][self.alphabet_size - 1] = sys.float_info.max

        return breakpoints

    def _igb(self, dft, y):
        breakpoints = np.zeros((self.word_length, self.alphabet_size))
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=np.log2(self.alphabet_size),
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

    def _mcb_dft(self, series, num_windows_per_inst):
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
            result[i] = self._discrete_fourier_transform(row)

        return result

    def _discrete_fourier_transform(self, series):
        """Performs a discrete fourier transform using the fast fourier
        transform
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
        dft[1::2] = imags[: np.uint32(length / 2)] * -1  # lower bounding
        dft *= self.inverse_sqrt_win_size / std
        return dft[start:]

    def _mft_start_length_end(self, series):
        start_offset = 2 if self.norm else 0
        length = self.dft_length + start_offset + self.dft_length % 2
        end = max(1, len(series) - self.window_size + 1)
        return start_offset, length, end

    # TODO: safe memory by directly discretizing and
    #       not storing the intermediate array?
    def _mft(self, series, transformed=None, stds=None):
        """

        :param series:
        :return:
        """
        start_offset, length, end = self._mft_start_length_end(series)

        phis = np.array(
            [
                [
                    math.cos(2 * math.pi * (-i) / self.window_size),
                    -math.sin(2 * math.pi * (-i) / self.window_size),
                ]
                for i in range(0, int(length / 2))
            ]
        ).flatten()

        if stds is None or np.shape(stds) != (end):
            stds = np.zeros(end)

        stds = SFA._calc_incremental_mean_std(series, end, self.window_size, stds)

        if transformed is None or np.shape(transformed) != (end, length):
            transformed = np.zeros((end, length))

        # first run with fft
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

        # lower bounding
        transformed[:, 1::2] = transformed[:, 1::2] * -1

        return (
            transformed[:, start_offset:][:, self.support]
            if self.anova
            else transformed[:, start_offset:]
        )

    @staticmethod
    @njit(
        # "(float64[:],float64[:],float64[:],int32,float64[:],float64[:,:],float64)",
        fastmath=True,
        cache=True,
    )
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

    # TODO merge with transform???
    # assumes saved words are of word length 'max_word_length'.
    def _shorten_bags(self, word_len):
        new_bags = pd.DataFrame() if self.return_pandas_data_series else [None]
        dim = []

        for i in range(len(self.words)):
            bag = defaultdict(int)
            # bag = bag = Dict.empty(key_type=typeof((100,100.0)),
            #                  value_type=types.float64) \
            #     if self.levels > 1 else \
            #     Dict.empty(key_type=types.int64, value_type=types.int64)

            last_word = -1
            repeat_words = 0

            for window, word in enumerate(self.words[i]):
                new_word = self.shorten_word(word, self.word_length - word_len)

                repeat_word = (
                    self._add_to_pyramid(
                        bag, new_word, last_word, window - int(repeat_words / 2)
                    )
                    if self.levels > 1
                    else self._add_to_bag(bag, new_word, last_word, window)
                )

                if repeat_word:
                    repeat_words += 1
                else:
                    last_word = new_word
                    repeat_words = 0

                if self.bigrams:
                    if window - self.window_size >= 0 and window > 0:
                        bigram = self.create_bigram_word(
                            self.shorten_word(
                                self.words[i][window - self.window_size],
                                self.word_length - word_len,
                            ),
                            new_word,
                            self.word_length,
                        )

                        if self.levels > 1:
                            bigram = (bigram, 0)
                        bag[bigram] += 1

            dim.append(pd.Series(bag) if self.return_pandas_data_series else bag)

        new_bags[0] = dim

        return new_bags

    def _add_to_bag(self, bag, word, last_word, offset):
        if self.remove_repeat_words and word == last_word:
            return False

        # store the histogram of word counts
        bag[word] += 1

        # store the first position of a word, too
        # bag[word << 16] = min(bag.get(word << 16,sys.float_info.max), offset)

        return True

    def _add_to_pyramid(self, bag, word, last_word, window_ind):
        if self.remove_repeat_words and word == last_word:
            return False

        start = 0
        for i in range(self.levels):
            num_quadrants = pow(2, i)
            quadrant_size = self.series_length / num_quadrants
            pos = window_ind + int((self.window_size / 2))
            quadrant = start + int(pos / quadrant_size)

            bag[word << 4 | quadrant] += self.level_weights[i]

            start += num_quadrants

        return True

    @staticmethod
    @njit(  # this seems to cause a problem with python 3.6??
        # "uint32(float64[:], int32, int32, float64[:,:])",
        # fastmath=True
        cache=True
    )
    def _create_word(dft, word_length, alphabet_size, breakpoints):
        word = np.int64(0)
        for i in range(word_length):
            for bp in range(alphabet_size):
                if dft[i] <= breakpoints[i][bp]:
                    word = (word << 2) | bp
                    break

        return word

    @staticmethod
    @njit(
        # "float64[:](float64[:],int64,int64,float64[:])",
        fastmath=True,
        cache=True,
    )
    def _calc_incremental_mean_std(series, end, window_size, stds=None):
        # means = np.zeros(end)

        if stds is None or len(stds) != end:
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

    # Used to represent a word for dictionary based classifiers such as BOSS
    # an BOP.
    # Can currently only handle an alphabet size of <= 4 and word length of
    # <= 16.
    # Current literature shows little reason to go beyond this, but the
    # class will need changes/expansions
    # if this is needed.
    # TODO a shift of 2 is only correct for alphabet size 4, log2(4)=2

    @staticmethod
    @njit("int64(int64,int64,int64)", fastmath=True, cache=True)
    def create_bigram_word(word, other_word, length):
        return (word << (2 * length)) | other_word

    # @staticmethod
    # @njit("int64(int64,int64,int64,int64)", fastmath=True, cache=True)
    # def create_skipgram_word(word, other_word, skips, length):
    #     return (((word << (2 * length)) | other_word) << 3) | skips

    @classmethod
    def shorten_word(cls, word, amount):
        # shorten a word by set amount of letters
        return cls.right_shift(word, amount * 2)

    @classmethod
    def word_list(cls, word, length):
        # list of input integers to obtain current word
        word_list = []
        shift = 32 - (length * 2)

        for _ in range(length - 1, -1, -1):
            word_list.append(cls.right_shift(word << shift, 32 - 2))
            shift += 2

        return word_list

    @staticmethod
    @njit(fastmath=True, cache=True)
    def right_shift(left, right):
        return (left % 0x100000000) >> right
