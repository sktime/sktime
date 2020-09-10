__author__ = ["Matthew Middlehurst"]
__all__ = ["SFA"]

import math
import sys

import numpy as np
import pandas as pd

from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.dictionary_based._sax import \
    _BitWord
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif

from numba import njit

# The binning methods to use: equi-depth, equi-width or information gain
binning_methods = {"equi-depth", "equi-width", "information-gain"}


@njit()
def _create_word(dft, word_length, alphabet_size, breakpoints):
    word = 0
    for i in range(word_length):
        for bp in range(alphabet_size):
            if dft[i] <= breakpoints[i][bp]:
                word = (word << 2) | bp
                break

    return word


@njit()
def _iterate_mft(series, mft_data, phis, length, window_size, i):
    for n in range(0, length, 2):
        real = mft_data[n] + series[i + window_size - 1] - \
               series[i - 1]
        imag = mft_data[n + 1]
        mft_data[n] = real * phis[n] - imag * phis[n + 1]
        mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag


@njit()
def _calc_incremental_mean_std(series, end, window_size):
    means = np.zeros(end)
    stds = np.zeros(end)

    window = series[0:window_size]
    series_sum = np.sum(window)
    square_sum = np.sum(np.multiply(window, window))

    r_window_length = 1 / window_size
    means[0] = series_sum * r_window_length
    buf = square_sum * r_window_length - means[0] * means[0]
    stds[0] = math.sqrt(buf) if buf > 0 else 0

    for w in range(1, end):
        series_sum += series[w + window_size - 1] - series[w - 1]
        means[w] = series_sum * r_window_length
        square_sum += series[w + window_size - 1] * series[
            w + window_size - 1] - series[w - 1] * series[w - 1]
        buf = square_sum * r_window_length - means[w] * means[w]
        stds[w] = math.sqrt(buf) if buf > 0 else 0

    return stds


class SFA(BaseSeriesAsFeaturesTransformer):
    """ SFA (Symbolic Fourier Approximation) Transformer, as described in

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

        norm:               boolean, default = False
            mean normalise words by dropping first fourier coefficient

        binning_method:      {"equi-depth", "equi-width", "information-gain"},
                             default="equi-depth"
            the binning method used to derive the breakpoints.

        anova:               boolean, default = False
            If True, the Fourier coefficient selection is done via a one-way
            ANOVA test. If False, the first Fourier coefficients are selected.
            Only applicable if labels are given

        bigrams:             boolean, default = False
            whether to create bigrams

        remove_repeat_words: boolean, default = False
            whether to use numerosity reduction (default False)

        save_words:          boolean, default = False
            whether to save the words generated for each series (default False)

    Attributes
    ----------
        words: []
        breakpoints: = []
        num_insts = 0
        num_atts = 0
    """

    def __init__(self,
                 word_length=8,
                 alphabet_size=4,
                 window_size=12,
                 norm=False,
                 binning_method="equi-depth",
                 anova=False,
                 bigrams=False,
                 remove_repeat_words=False,
                 levels=1,
                 save_words=False
                 ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        offset = 2 if norm else 0
        self.word_length = min(word_length, window_size - offset)
        self.dft_length = window_size - offset if anova is True \
            else self.word_length

        # make dft_length an even number (same number of reals and imags)
        self.dft_length = self.dft_length + self.dft_length % 2

        self.support = list(range(self.word_length))

        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.inverse_sqrt_win_size = 1.0 / math.sqrt(window_size)
        self.norm = norm
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words

        # TDE
        self.levels = levels

        #
        self.binning_method = binning_method
        self.anova = anova

        self.bigrams = bigrams
        # weighting for levels going up to 7 levels
        # No real reason to go past 3
        self.level_weights = [1, 2, 4, 16, 32, 64, 128]

        self.n_instances = 0
        self.series_length = 0
        super(SFA, self).__init__()

    def fit(self, X, y=None):
        """Calculate word breakpoints using _mcb

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The class labels.

        Returns
        -------
        self : object
         """

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise ValueError(
                "Alphabet size must be an integer between 2 and 4")

        if self.word_length < 1 or self.word_length > 16:
            raise ValueError(
                "Word length must be an integer between 1 and 16")

        if self.binning_method == "information-gain" and y is None:
            raise ValueError(
                "Class values must be provided for information gain binning")

        if self.binning_method not in binning_methods:
            raise TypeError('binning_method must be one of: ', binning_methods)

        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)

        self.n_instances, self.series_length = X.shape
        self.breakpoints = self._binning(X, y)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)

        bags = pd.DataFrame()
        dim = []

        for i in range(X.shape[0]):
            dfts = self._mft(X[i, :])
            bag = {}
            last_word = -1
            repeat_words = 0
            words = []

            for window in range(dfts.shape[0]):
                word_raw = _create_word(
                    dfts[window], self.word_length,
                    self.alphabet_size, self.breakpoints)
                word = _BitWord(word=word_raw)

                words.append(word)
                repeat_word = (self._add_to_pyramid(bag, word, last_word,
                                                    window -
                                                    int(repeat_words / 2))
                               if self.levels > 1 else
                               self._add_to_bag(bag, word, last_word))
                if repeat_word:
                    repeat_words += 1
                else:
                    last_word = word.word
                    repeat_words = 0

                if self.bigrams:
                    if window - self.window_size >= 0 and window > 0:
                        bigram = words[window - self.window_size] \
                            .create_bigram(word, self.word_length)
                        if self.levels > 1:
                            bigram = (bigram, 0)
                        bag[bigram] = bag.get(bigram, 0) + 1

            if self.save_words:
                self.words.append(words)

            dim.append(pd.Series(bag))

        bags[0] = dim

        return bags

    def _binning(self, X, y=None):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        dft = np.array([self._mcb_dft(X[i, :], num_windows_per_inst) for i in
                        range(self.n_instances)])
        dft = dft.reshape(len(X) * num_windows_per_inst, self.dft_length)

        if y is not None:
            y = np.repeat(y, num_windows_per_inst)

        if self.anova and y is not None:
            # non_constant = np.where(~np.isclose(
            #                dft.var(axis=0), np.zeros_like(dft.shape[1])))[0]
            _, p = f_classif(dft, y)
            # self.support=non_constant[np.argsort(p)[::-1][:self.word_length]]
            # select word-length many indices with largest f-score
            self.support = np.argsort(p)[::-1][:self.word_length]

            # sort remaining indices
            self.support = np.sort(self.support)

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support)
            self.dft_length = self.dft_length + self.dft_length % 2  # even

        if self.binning_method == "information-gain":
            return self._igb(X, y, dft)
        else:
            return self._mcb(X, y, dft)

    def _mcb(self, X, y, dft):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        total_num_windows = self.n_instances * num_windows_per_inst
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):

            res = [round(dft[inst][letter] * 100) / 100
                   for inst in range(self.n_instances * num_windows_per_inst)]
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
                target_bin_width = \
                    (column[-1] - column[0]) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    breakpoints[letter][bp] = (bp + 1) * target_bin_width \
                                              + column[0]

            breakpoints[letter][self.alphabet_size - 1] = sys.float_info.max

        return breakpoints

    def _igb(self, X, y, dft):
        breakpoints = np.zeros((self.word_length, self.alphabet_size))
        clf = DecisionTreeClassifier(criterion='entropy',
                                     max_leaf_nodes=self.alphabet_size)

        for i in range(self.word_length):
            clf.fit(dft[:, i][:, None], y)
            threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
            for bp in range(len(threshold)):
                breakpoints[i][bp] = threshold[bp]

            breakpoints[i][self.alphabet_size - 1] = sys.float_info.max
        return np.sort(breakpoints, axis=1)

    def _mcb_dft(self, series, num_windows_per_inst):
        # Splits individual time series into windows and returns the DFT for
        # each
        split = np.split(series, np.linspace(self.window_size,
                                             self.window_size * (
                                                     num_windows_per_inst - 1),
                                             num_windows_per_inst - 1,
                                             dtype=np.int_))
        split[-1] = series[self.series_length -
                           self.window_size:self.series_length]
        return [self._discrete_fourier_transform(row) for n, row in
                enumerate(split)]

    def _discrete_fourier_transform(self, series):
        """ Performs a discrete fourier transform using the fast fourier
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
        std = (s if s > 0 else 1)

        X_fft = np.fft.rfft(series)
        reals = np.real(X_fft)
        imags = np.imag(X_fft)  # * -1 # TODO correct for lower bounding??

        length = start + self.dft_length
        dft = np.empty((length,), dtype=reals.dtype)
        dft[0::2] = reals[:np.int32(length / 2)]
        dft[1::2] = imags[:np.int32(length / 2)]
        dft *= self.inverse_sqrt_win_size / std
        return dft[start:]

    def _mft(self, series):
        """

        :param series:
        :return:
        """
        start_offset = 2 if self.norm else 0
        length = self.dft_length + start_offset + self.dft_length % 2

        phis = np.array([[
            math.cos(2 * math.pi * (-i) / self.window_size),
            -math.sin(2 * math.pi * (-i) / self.window_size)]
            for i in range(0, int(length / 2))]).flatten()

        end = max(1, len(series) - self.window_size + 1)
        stds = _calc_incremental_mean_std(series, end, self.window_size)
        transformed = np.zeros((end, length))
        mft_data = np.array([])

        for i in range(end):
            if i > 0:
                # moved to external method to use njit
                # for n in range(0, length, 2):
                #     real = mft_data[n] + series[i + self.window_size - 1] - \
                #            series[i - 1]
                #     imag = mft_data[n + 1]
                #     mft_data[n] = real * phis[n] - imag * phis[n + 1]
                #     mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag
                _iterate_mft(series, mft_data, phis, length,
                             self.window_size, i)
            else:
                X_fft = np.fft.rfft(series[0:self.window_size])
                reals = np.real(X_fft)
                imags = np.imag(X_fft)  # * -1 # TODO lower bounding??
                mft_data = np.empty((length,), dtype=reals.dtype)
                mft_data[0::2] = reals[:np.int32(length / 2)]
                mft_data[1::2] = imags[:np.int32(length / 2)]

            normalising_factor = ((1 / stds[i] if stds[i] > 0 else 1) *
                                  self.inverse_sqrt_win_size)

            transformed[i] = mft_data * normalising_factor

        return transformed[:, start_offset:][:, self.support] \
            if self.anova else transformed[:, start_offset:]

    # moved to external method to use njit
    # def _calc_incremental_mean_std(self, series, end):
    #     means = np.zeros(end)
    #     stds = np.zeros(end)
    #
    #     window = series[0:self.window_size]
    #     series_sum = np.sum(window)
    #     square_sum = np.sum(np.multiply(window, window))
    #
    #     r_window_length = 1 / self.window_size
    #     means[0] = series_sum * r_window_length
    #     buf = square_sum * r_window_length - means[0] * means[0]
    #     stds[0] = math.sqrt(buf) if buf > 0 else 0
    #
    #     for w in range(1, end):
    #         series_sum += series[w + self.window_size - 1] - series[w - 1]
    #         means[w] = series_sum * r_window_length
    #         square_sum += series[w + self.window_size - 1] * series[
    #             w + self.window_size - 1] - series[w - 1] * series[w - 1]
    #         buf = square_sum * r_window_length - means[w] * means[w]
    #         stds[w] = math.sqrt(buf) if buf > 0 else 0
    #
    #     return stds
    #
    # def _create_word(self, dft):
    #     word = _BitWord()
    #
    #     for i in range(self.word_length):
    #         for bp in range(self.alphabet_size):
    #             if dft[i] <= self.breakpoints[i][bp]:
    #                 word.push(bp)
    #                 break
    #
    #     return word

    # assumes saved words are of word length 16.
    def _shorten_bags(self, word_len):
        new_bags = pd.DataFrame()
        dim = []

        for i in range(len(self.words)):
            bag = {}
            last_word = -1
            repeat_words = 0
            new_words = []
            for window, word in enumerate(self.words[i]):
                new_word = _BitWord(word=word.word)
                new_word.shorten(16 - word_len)
                repeat_word = (self._add_to_pyramid(bag, new_word, last_word,
                                                    window -
                                                    int(repeat_words / 2))
                               if self.levels > 1 else
                               self._add_to_bag(bag, new_word, last_word))
                if repeat_word:
                    repeat_words += 1
                else:
                    last_word = new_word.word
                    repeat_words = 0

                if self.bigrams:
                    new_words.append(new_words)

                    if window - self.window_size >= 0 and window > 0:
                        bigram = new_words[window - self.window_size] \
                            .create_bigram(word, self.word_length)
                        if self.levels > 1:
                            bigram = (bigram, 0)
                        bag[bigram] = bag.get(bigram, 0) + 1

            dim.append(pd.Series(bag))

        new_bags[0] = dim

        return new_bags

    def _add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word.word == last_word:
            return False

        bag[word.word] = bag.get(word.word, 0) + 1

        return True

    def _add_to_pyramid(self, bag, word, last_word, window_ind):
        if self.remove_repeat_words and word.word == last_word:
            return False

        start = 0
        for i in range(self.levels):
            num_quadrants = pow(2, i)
            quadrant_size = self.series_length / num_quadrants
            pos = window_ind + int((self.window_size / 2))
            quadrant = start + (pos / quadrant_size)

            bag[(word.word, quadrant)] = (bag.get((word.word, quadrant), 0)
                                          + self.level_weights[i])

            start += num_quadrants

        return True
