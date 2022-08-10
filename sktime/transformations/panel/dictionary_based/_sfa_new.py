# -*- coding: utf-8 -*-
"""Symbolic Fourier Approximation (SFA) Transformer.

Configurable SFA transform for discretising time series into words.
"""

__author__ = ["Patrick Schäfer"]
__all__ = ["SFA_NEW"]

import math
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from numba import NumbaTypeSafetyWarning, njit
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X

# The binning methods to use: equi-depth, equi-width, information gain or kmeans
binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans"}

# TODO remove imag-part from dc-component component


class SFA_NEW(_PanelToPanelTransformer):
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

        variance:               boolean, default = False
            If True, the Fourier coefficient selection is done via a the largest
            variance. If False, the first Fourier coefficients are selected.
            Only applicable if labels are given

        bigrams:             boolean, default = False
            whether to create bigrams of SFA words

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

    _tags = {"univariate-only": True}

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
        lower_bounding=True,
        n_jobs=1,
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        offset = 2 if norm else 0
        self.dft_length = (
            window_size - offset if (anova or variance) is True else word_length
        )
        # make dft_length an even number (same number of reals and imags)
        self.dft_length = self.dft_length + self.dft_length % 2

        self.support = np.array(list(range(word_length)))

        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.lower_bounding = lower_bounding
        self.inverse_sqrt_win_size = (
            1.0 / math.sqrt(window_size) if not lower_bounding else 1.0
        )

        self.norm = norm

        self.binning_dft = None

        self.binning_method = binning_method
        self.anova = anova
        self.variance = variance

        self.bigrams = bigrams
        self.n_jobs = n_jobs

        self.n_instances = 0
        self.series_length = 0

        self.letter_bits = 0
        self.word_bits = 0
        self.max_bits = 0

        super(SFA_NEW, self).__init__()

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

        if self.binning_method == "information-gain" and y is None:
            raise ValueError(
                "Class values must be provided for information gain binning"
            )

        if self.variance and self.anova:
            raise ValueError("Please set either variance or anova feature selection")

        if self.binning_method not in binning_methods:
            raise TypeError("binning_method must be one of: ", binning_methods)

        self.letter_bits = math.ceil(math.log2(self.alphabet_size))
        self.word_bits = self.word_length * self.letter_bits
        self.max_bits = self.word_bits * 2 if self.bigrams else self.word_bits

        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        self.n_instances, self.series_length = X.shape
        self.breakpoints = self._binning(X, y)

        # TODO parameterize?
        self.breakpoints[self.breakpoints < 0] = -np.inf

        self._is_fitted = True
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
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        # self.breakpoints[self.breakpoints > 0] = np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)
            transform = Parallel(n_jobs=self.n_jobs)(
                delayed(self._transform_case)(X[i, :]) for i in range(X.shape[0])
            )

        _, words = zip(*transform)
        return words

    def _transform_case(self, X):
        dfts = self._mft(X)

        words = []

        for window in range(dfts.shape[0]):
            word_raw = SFA_NEW._create_word(
                dfts[window],
                self.word_length,
                self.alphabet_size,
                self.breakpoints,
                self.letter_bits,
            )
            words.append(word_raw)

            if self.bigrams:
                if window - self.window_size >= 0:
                    bigram = self._create_bigram_words(
                        word_raw, words[window - self.window_size]
                    )
                    words.append(bigram)

        return [{}, np.array(words)]

    def _binning(self, X, y=None):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        dft = np.array(
            [
                self._binning_dft(X[i, :], num_windows_per_inst)
                for i in range(self.n_instances)
            ]
        )

        dft = dft.reshape(len(X) * num_windows_per_inst, self.dft_length)

        if y is not None:
            y = np.repeat(y, num_windows_per_inst)

        if self.variance and y is not None:
            # determine variance
            dft_variance = dft.var(axis=0)

            # select word-length-many indices with largest variance
            self.support = np.argsort(-dft_variance)[: self.word_length]

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support) + 1
            self.dft_length = self.dft_length + self.dft_length % 2  # even

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
            result[i] = self._fast_fourier_transform(row)

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

    def _mft(self, series):
        start_offset = 2 if self.norm else 0
        length = self.dft_length + start_offset + self.dft_length % 2
        end = max(1, len(series) - self.window_size + 1)

        phis = SFA_NEW._get_phis(self.window_size, length)
        stds = SFA_NEW._calc_incremental_mean_std(series, end, self.window_size)
        transformed = np.zeros((end, length))

        X_fft = np.fft.rfft(series[: self.window_size])
        reals = np.real(X_fft)
        imags = np.imag(X_fft)
        mft_data = np.empty((length,), dtype=reals.dtype)
        mft_data[0::2] = reals[: np.uint32(length / 2)]
        mft_data[1::2] = imags[: np.uint32(length / 2)]

        transformed[0] = mft_data * self.inverse_sqrt_win_size / stds[0]

        # other runs using mft
        # moved to external method to use njit
        SFA_NEW._iterate_mft(
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
            if (self.anova or self.variance)
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
            SFA_NEW._create_bigram_word(
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
