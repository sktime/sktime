import numpy as np
import pandas as pd
import math
import sys

from sktime.utils.bitword import BitWord
from sktime.transformers.base import BaseTransformer


class SFA(BaseTransformer):
    """
    SFA Transform for a fixed set of parameters
    By default returns the word for each input instance
    Options allows for configurations used in BOSS and related classifiers
    """

    def __init__(self,
                 word_length,
                 alphabet_size,
                 window_size=0,
                 norm=False,
                 remove_repeat_words=False,
                 dim_to_use=0,
                 save_words=False
                 ):
        self.words = []
        self.breakpoints = []

        self.word_length = word_length
        self.alphabet_size = alphabet_size

        self.window_size = window_size
        if window_size != 0:
            self.inverse_sqrt_win_size = 1 / math.sqrt(window_size)

        self.norm = norm
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words

        # For the multivariate case treating this as a univariate classifier
        self.dim_to_use = dim_to_use

        self.num_insts = 0
        self.num_atts = 0

    def fit(self, X, **kwargs):
        """Build a histogram

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]

        Returns
        -------
        self : object
        :param **kwargs:
         """

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise RuntimeError("Alphabet size must be an integer between 2 and 4")

        if self.word_length < 1 or self.word_length > 16:
            raise RuntimeError("Word length must be an integer between 1 and 16")

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing "
                                "Series objects")

        self.num_insts, self.num_atts = X.shape

        if self.window_size == 0:
            self.window_size = self.num_atts
            self.inverse_sqrt_win_size = 1 / math.sqrt(self.window_size)

        self.breakpoints = self.MCB(X)

        self.is_fitted_ = True

        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("The fit method must be called before calling transform")

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe containing "
                                "Series objects")

        self.num_insts = X.shape[0]

        bags = pd.DataFrame()
        dim = []

        for i in range(self.num_insts):
            dfts = self.MFT(X[i, :])
            bag = {}
            lastWord = None

            words = []

            for window in range(dfts.shape[0]):
                word = self.create_word(dfts[window])
                words.append(word)
                lastWord = self.add_to_bag(bag, word, lastWord)

            if self.save_words:
                self.words.append(words)

            dim.append(pd.Series(bag))

        bags['dim_' + str(self.dim_to_use)] = dim

        return bags

    def MCB(self, X):
        """


        :param X: the training data
        :return:
        """
        num_windows_per_inst = math.ceil(self.num_atts / self.window_size)
        dft = np.zeros((self.num_insts, num_windows_per_inst, int((self.word_length / 2) * 2)))

        for i in range(self.num_insts):
            split = np.split(X[i, :], np.linspace(self.window_size, self.window_size * (num_windows_per_inst - 1),
                                                  num_windows_per_inst - 1, dtype=np.int_))
            split[-1] = X[i, self.num_atts - self.window_size:self.num_atts]

            for n, row in enumerate(split):
                dft[i, n] = self.discrete_fourier_transform(row)

        total_num_windows = self.num_insts * num_windows_per_inst
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):
            column = np.zeros(total_num_windows)

            for inst in range(self.num_insts):
                for window in range(num_windows_per_inst):
                    column[(inst * num_windows_per_inst) + window] = round(dft[inst][window][letter] * 100) / 100

            column = np.sort(column)

            bin_index = 0
            target_bin_depth = total_num_windows / self.alphabet_size

            for bp in range(self.alphabet_size - 1):
                bin_index += target_bin_depth
                breakpoints[letter][bp] = column[int(bin_index)]

            breakpoints[letter][self.alphabet_size - 1] = sys.float_info.max

        return breakpoints

    def discrete_fourier_transform(self, series, normalise=True):
        """ Performs a discrete fourier transform using standard O(n^2) transform
        if self.norm is True, then the first term of the DFT is ignored

        TO DO: Use a fast fourier transform
        Input
        -------
        X : The training input samples.  array-like or sparse matrix of shape = [n_samps, num_atts]

        Returns
        -------
        1D array of fourier term, real_0,imag_0, real_1, imag_1 etc, length num_atts or
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

        # dft2 = np.array([np.sum([series[n] * math.cos(2 * math.pi * n * i / length) for n in range(length)]) for i in
        #                  range(start, start + output_length)])
        # print(dft2)
        #
        # dft2 = np.array([np.sum([-series[n] * math.sin(2 * math.pi * n * i / length) for n in range(length)]) for i in
        #                  range(start, start + output_length)])
        # print(dft2)
        #

        dft = np.zeros(output_length * 2)

        for i in range(start, start + output_length):
            idx = (i - start) * 2

            for n in range(length):
                dft[idx] += series[n] * math.cos(2 * math.pi * n * i / length)
                dft[idx + 1] += -series[n] * math.sin(2 * math.pi * n * i / length)

        # print(dft)

        if normalise:
            dft *= self.inverse_sqrt_win_size / std

        return dft

    def MFT(self, series):
        """

        :param series:
        :return:
        """
        start_offset = 2 if self.norm else 0
        length = self.word_length + self.word_length % 2
        phis = np.zeros(length)

        for i in range(0, length, 2):
            half = -(i + start_offset) / 2
            phis[i] = math.cos(2 * math.pi * half / self.window_size)
            phis[i + 1] = -math.sin(2 * math.pi * half / self.window_size)

        end = max(1, len(series) - self.window_size + 1)
        stds = self.calc_incremental_mean_std(series, end)
        transformed = np.zeros((end, length))
        mft_data = None

        for i in range(end):
            if i > 0:
                for n in range(0, length, 2):
                    real1 = mft_data[n] + series[i + self.window_size - 1] - series[i - 1]
                    imag1 = mft_data[n + 1]
                    real = real1 * phis[n] - imag1 * phis[n + 1]
                    imag = real1 * phis[n + 1] + phis[n] * imag1
                    mft_data[n] = real
                    mft_data[n + 1] = imag
            else:
                mft_data = self.discrete_fourier_transform(series[0:self.window_size], normalise=False)

            normalising_factor = (1 / stds[i] if stds[i] > 0 else 1) * self.inverse_sqrt_win_size
            transformed[i] = mft_data * normalising_factor

        return transformed

    def calc_incremental_mean_std(self, series, end):
        means = np.zeros(end)
        stds = np.zeros(end)

        series_sum = 0
        square_sum = 0

        for ww in range(self.window_size):
            series_sum += series[ww]
            square_sum += series[ww] * series[ww]

        rWindowLength = 1 / self.window_size
        means[0] = series_sum * rWindowLength
        buf = square_sum * rWindowLength - means[0] * means[0]
        stds[0] = math.sqrt(buf) if buf > 0 else 0

        for w in range(1, end):
            series_sum += series[w + self.window_size - 1] - series[w - 1]
            means[w] = series_sum * rWindowLength
            square_sum += series[w + self.window_size - 1] * series[w + self.window_size - 1] - series[w - 1] * series[
                w - 1]
            buf = square_sum * rWindowLength - means[w] * means[w]
            stds[w] = math.sqrt(buf) if buf > 0 else 0

        return stds

    def create_word(self, dft):
        word = BitWord()

        for i in range(self.word_length):
            for bp in range(self.alphabet_size):
                if dft[i] <= self.breakpoints[i][bp]:
                    word.push(bp)
                    break

        return word

    def shorten_bags(self, word_len):
        new_bags = pd.DataFrame()
        dim = []

        for i in range(self.num_insts):
            bag = {}
            last_word = -1

            for n, word in enumerate(self.words[i]):
                new_word = BitWord(word=word.word, length=word.length)
                new_word.shorten(16 - word_len)
                last_word = self.add_to_bag(bag, new_word, last_word)

            dim.append(pd.Series(bag))

        new_bags['dim_' + str(self.dim_to_use)] = dim

        return new_bags

    def add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word.word == last_word:
            return last_word

        if word.word in bag:
            bag[word.word] += 1
        else:
            bag[word.word] = 1

        return word.word
