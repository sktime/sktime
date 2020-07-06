__author__ = ["Matthew Middlehurst"]
__all__ = ["SFA"]

import math
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.dictionary_based._sax import \
    _BitWord
from sktime.utils.data_container import tabularize
from sktime.utils.validation.series_as_features import check_X


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
        word_length:         int, length of word to shorten window to (using
        PAA) (default 8)
        alphabet_size:       int, number of values to discretise each value
        to (default to 4)
        window_size:         int, size of window for sliding. Input series
        length for whole series transform (default to 12)
        norm:                boolean, whether to mean normalise words by
        dropping first fourier coefficient
        remove_repeat_words: boolean, whether to use numerosity reduction (
        default False)
        save_words:          boolean, whether to save the words generated
        for each series (default False)

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
                 levels=1,
                 igb=False,
                 bigrams=False,
                 remove_repeat_words=False,
                 save_words=False
                 ):
        self.words = []
        self.breakpoints = []

        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.inverse_sqrt_win_size = 1 / math.sqrt(window_size)
        self.norm = norm
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words

        # TDE
        self.levels = levels
        self.igb = igb
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

        if self.igb and y is None:
            raise ValueError(
                "Class values must be provided for information gain binning")

        X = check_X(X, enforce_univariate=True)
        X = tabularize(X, return_array=True)

        self.n_instances, self.series_length = X.shape
        self.breakpoints = self._igb(X, y) if self.igb else self._mcb(X)

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
                word = self._create_word(dfts[window])
                words.append(word)
                repeat_word = (self._add_to_pyramid(bag, word, last_word,
                                                    window -
                                                    int(repeat_words/2))
                               if self.levels > 1 else
                               self._add_to_bag(bag, word, last_word))
                if repeat_word:
                    repeat_words += 1
                else:
                    last_word = word.word
                    repeat_words = 0

                if self.bigrams:
                    if window - self.window_size >= 0 and window > 0:
                        bigram = words[window - self.window_size]\
                            .create_bigram(word, self.word_length)
                        if self.levels > 1:
                            bigram = (bigram, 0)
                        bag[bigram] = bag.get(bigram, 0) + 1

            if self.save_words:
                self.words.append(words)

            dim.append(pd.Series(bag))

        bags[0] = dim

        return bags

    def _mcb(self, X):
        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        dft = np.array([self._mcb_dft(X[i, :], num_windows_per_inst) for i in
                        range(self.n_instances)])

        total_num_windows = self.n_instances * num_windows_per_inst
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):
            column = np.sort(
                np.array([round(dft[inst][window][letter] * 100) / 100
                          for window in range(num_windows_per_inst) for inst in
                          range(self.n_instances)]))

            bin_index = 0
            target_bin_depth = total_num_windows / self.alphabet_size

            for bp in range(self.alphabet_size - 1):
                bin_index += target_bin_depth
                breakpoints[letter][bp] = column[int(bin_index)]

            breakpoints[letter][self.alphabet_size - 1] = sys.float_info.max

        return breakpoints

    def _igb(self, X, y):
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        num_windows_per_inst = math.ceil(self.series_length / self.window_size)
        dft = np.array([self._igb_dft(X[i, :], num_windows_per_inst, y[i])
                        for i in range(self.n_instances)])

        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):
            column = [(round(dft[inst][window][letter][0] * 100) / 100,
                       dft[inst][window][letter][1])
                      for window in range(num_windows_per_inst)
                      for inst in range(self.n_instances)]
            column.sort(key=lambda tup: tup[0])

            splits = []
            self._find_split_points(column, 0, len(column),
                                    self.alphabet_size, splits)
            splits.sort()

            for bp in range(len(splits)):
                breakpoints[letter][bp] = column[splits[bp]][0]

            breakpoints[letter][self.alphabet_size - 1] = sys.float_info.max

        return breakpoints

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

    def _igb_dft(self, series, num_windows_per_inst, cls):
        # Splits individual time series into windows and returns a DFT and
        # class pair for each
        split = np.split(series, np.linspace(self.window_size,
                                             self.window_size * (
                                                     num_windows_per_inst - 1),
                                             num_windows_per_inst - 1,
                                             dtype=np.int_))
        split[-1] = series[self.series_length -
                           self.window_size:self.series_length]
        return [[(i, cls) for i in self._discrete_fourier_transform(row)]
                for n, row in enumerate(split)]

    def _discrete_fourier_transform(self, series, normalise=True):
        """ Performs a discrete fourier transform using standard O(n^2)
        transform
        if self.norm is True, then the first term of the DFT is ignored

        TO DO: Use a fast fourier transform
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

        length = len(series)
        output_length = int(self.word_length / 2)
        start = 1 if self.norm else 0

        std = 1
        if normalise:
            s = np.std(series)
            if s != 0:
                std = s

        dft = np.array(
            [np.sum([[series[n] * math.cos(2 * math.pi * n * i / length),
                      -series[n] * math.sin(2 * math.pi * n * i / length)] for
                     n in range(length)], axis=0)
             for i in range(start, start + output_length)]).flatten()

        if normalise:
            dft *= self.inverse_sqrt_win_size / std

        return dft

    def _find_split_points(self, points, start, end, remaining_symbols,
                           splits):
        out_dict = {}
        in_dict = {}
        for p in range(start, end):
            out_dict[points[p][1]] = out_dict.get(points[p][1], 0) + 1

        class_entropy = self._entropy(out_dict)

        last_label = points[start][1]
        out_dict[points[start][1]] = out_dict.get(points[start][1], 0) - 1
        in_dict[points[start][1]] = in_dict.get(points[start][1], 0) + 1

        best_gain = -1
        best_pos = -1

        for i in range(start + 1, end - 1):
            label = points[i][1]
            out_dict[points[i][1]] = out_dict.get(points[i][1], 0) - 1
            in_dict[points[i][1]] = in_dict.get(points[i][1], 0) + 1

            if label != last_label:
                gain = (round(self._information_gain(class_entropy, in_dict,
                                                     out_dict) * 1000) / 1000)

                if gain >= best_gain:
                    best_gain = gain
                    best_pos = i

            last_label = label

        if best_pos > -1:
            splits.append(best_pos)

            remaining_symbols /= 2
            if remaining_symbols > 1:
                if best_pos - start > 2 and end - best_pos > 2:
                    self._find_split_points(points, start, best_pos,
                                            remaining_symbols, splits)
                    self._find_split_points(points, best_pos, end,
                                            remaining_symbols, splits)
                elif end - best_pos > 4:
                    self._find_split_points(points, best_pos,
                                            int((end - best_pos) / 2),
                                            remaining_symbols, splits)
                    self._find_split_points(points,
                                            int((end - best_pos) / 2),
                                            end, remaining_symbols, splits)
                elif best_pos - start > 4:
                    self._find_split_points(points, start,
                                            int((best_pos - start) / 2),
                                            remaining_symbols, splits)
                    self._find_split_points(points,
                                            int((best_pos - start) / 2),
                                            end, remaining_symbols, splits)

        return splits

    def _entropy(self, frequency_dict, total=-1):
        if total == -1:
            total = sum(frequency_dict.values())
        log2 = 1.0 / math.log(2.0)
        entropy = 0
        for i in frequency_dict.values():
            p = i/total
            if p > 0:
                entropy -= p * math.log(p) * log2
        return entropy

    def _information_gain(self, class_entropy, in_freq_dict, out_freq_dict):
        in_total = sum(in_freq_dict.values())
        out_total = sum(out_freq_dict.values())
        total = in_total + out_total
        return (class_entropy
                - in_total / total * self._entropy(in_freq_dict, in_total)
                - out_total / total * self._entropy(out_freq_dict, out_total))

    def _mft(self, series):
        """

        :param series:
        :return:
        """
        start_offset = 2 if self.norm else 0
        length = self.word_length + self.word_length % 2

        phis = np.array([[math.cos(
            2 * math.pi * (-((i * 2) + start_offset) / 2) / self.window_size),
            -math.sin(2 * math.pi * (-((
                                               i * 2) +
                                       start_offset) / 2) /
                      self.window_size)]
            for i in range(0, int(length / 2))]).flatten()

        end = max(1, len(series) - self.window_size + 1)
        stds = self._calc_incremental_mean_std(series, end)
        transformed = np.zeros((end, length))
        mft_data = np.array([])

        for i in range(end):
            if i > 0:
                for n in range(0, length, 2):
                    real = mft_data[n] + series[i + self.window_size - 1] - \
                           series[i - 1]
                    imag = mft_data[n + 1]
                    mft_data[n] = real * phis[n] - imag * phis[n + 1]
                    mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag
            else:
                mft_data = self._discrete_fourier_transform(
                    series[0:self.window_size], normalise=False)

            normalising_factor = ((1 / stds[i] if stds[i] > 0 else 1) *
                                  self.inverse_sqrt_win_size)
            transformed[i] = mft_data * normalising_factor

        return transformed

    def _calc_incremental_mean_std(self, series, end):
        means = np.zeros(end)
        stds = np.zeros(end)

        window = series[0:self.window_size]
        series_sum = np.sum(window)
        square_sum = np.sum(np.multiply(window, window))

        r_window_length = 1 / self.window_size
        means[0] = series_sum * r_window_length
        buf = square_sum * r_window_length - means[0] * means[0]
        stds[0] = math.sqrt(buf) if buf > 0 else 0

        for w in range(1, end):
            series_sum += series[w + self.window_size - 1] - series[w - 1]
            means[w] = series_sum * r_window_length
            square_sum += series[w + self.window_size - 1] * series[
                w + self.window_size - 1] - series[w - 1] * series[
                              w - 1]
            buf = square_sum * r_window_length - means[w] * means[w]
            stds[w] = math.sqrt(buf) if buf > 0 else 0

        return stds

    def _create_word(self, dft):
        word = _BitWord()

        for i in range(self.word_length):
            for bp in range(self.alphabet_size):
                if dft[i] <= self.breakpoints[i][bp]:
                    word.push(bp)
                    break

        return word

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
                                                    int(repeat_words/2))
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
