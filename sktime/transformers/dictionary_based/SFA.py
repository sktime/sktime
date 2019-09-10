import numpy as np
import pandas as pd
import math
import sys

from sktime.transformers.dictionary_based.SAX import BitWord
from sktime.transformers.base import BaseTransformer


# TO DO: Finish comments


class SFA(BaseTransformer):
    __author__ = "Matthew Middlehurst"
    """ SFA Transformer, as described in 

    @inproceedings{schafer2012sfa,
      title={SFA: a symbolic fourier approximation and index for similarity search in high dimensional datasets},
      author={Sch{\"a}fer, Patrick and H{\"o}gqvist, Mikael},
      booktitle={Proceedings of the 15th International Conference on Extending Database Technology},
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
        word_length:         int, length of word to shorten window to (using PAA) (default 8)
        alphabet_size:       int, number of values to discretise each value to (default to 4)
        window_size:         int, size of window for sliding. If 0, uses the whole series (default to 0)
        norm:                boolean, whether to mean normalise words by dropping first fourier coefficient
        remove_repeat_words: boolean, whether to use numerosity reduction (default False)
        save_words:          boolean, whether to save the words generated for each series (default False)

    Attributes
    ----------
        words: []
        breakpoints: = []
        num_insts = 0
        num_atts = 0
"""

    def __init__(self,
                 word_length,
                 alphabet_size,
                 window_size=0,
                 norm=False,
                 remove_repeat_words=False,
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
        self.num_insts = 0
        self.num_atts = 0

    def fit(self, X, y=None):
        """Calculate word breakpoints using MCB

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts]
            The training input samples.  If a Pandas data frame is passed, the column _dim_to_use is extracted
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The class labels.

        Returns
        -------
        self : object
         """

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise RuntimeError("Alphabet size must be an integer between 2 and 4")

        if self.word_length < 1 or self.word_length > 16:
            raise RuntimeError("Word length must be an integer between 1 and 16")

        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("SFA cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0, 0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series "
                    "objects (TSF cannot yet handle multivariate problems")

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
            if X.shape[1] > 1:
                raise TypeError("SFA cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0, 0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series "
                    "objects (TSF cannot yet handle multivariate problems")

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

        bags['dim_' + str(0)] = dim

        return bags

    def MCB(self, X):
        """


        :param X: the training data
        :return:
        """
        num_windows_per_inst = math.ceil(self.num_atts / self.window_size)
        dft = np.array([self.MCB_DFT(X[i, :], num_windows_per_inst) for i in range(self.num_insts)])

        # dft = np.zeros((self.num_insts, num_windows_per_inst, int((self.word_length / 2) * 2)))
        #
        # for i in range(self.num_insts):
        #     split = np.split(X[i, :], np.linspace(self.window_size, self.window_size * (num_windows_per_inst - 1),
        #                                           num_windows_per_inst - 1, dtype=np.int_))
        #     split[-1] = X[i, self.num_atts - self.window_size:self.num_atts]
        #     dft[i] = np.array([self.discrete_fourier_transform(row) for n, row in enumerate(split)])

        total_num_windows = self.num_insts * num_windows_per_inst
        breakpoints = np.zeros((self.word_length, self.alphabet_size))

        for letter in range(self.word_length):
            column = np.sort(np.array([round(dft[inst][window][letter] * 100) / 100
                                       for window in range(num_windows_per_inst) for inst in range(self.num_insts)]))

            # column = np.zeros(total_num_windows)
            #
            # for inst in range(self.num_insts):
            #     for window in range(num_windows_per_inst):
            #         column[(inst * num_windows_per_inst) + window] = round(dft[inst][window][letter] * 100) / 100
            #
            # column = np.sort(column)

            bin_index = 0
            target_bin_depth = total_num_windows / self.alphabet_size

            for bp in range(self.alphabet_size - 1):
                bin_index += target_bin_depth
                breakpoints[letter][bp] = column[int(bin_index)]

            breakpoints[letter][self.alphabet_size - 1] = sys.float_info.max

        return breakpoints

    def MCB_DFT(self, series, num_windows_per_inst):
        # Splits individual time series into windows and returns the DFT for each
        split = np.split(series, np.linspace(self.window_size, self.window_size * (num_windows_per_inst - 1),
                                             num_windows_per_inst - 1, dtype=np.int_))
        split[-1] = series[self.num_atts - self.window_size:self.num_atts]
        return [self.discrete_fourier_transform(row) for n, row in enumerate(split)]

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

        dft = np.array([np.sum([[series[n] * math.cos(2 * math.pi * n * i / length),
                                 -series[n] * math.sin(2 * math.pi * n * i / length)] for n in range(length)], axis=0)
                        for i in range(start, start + output_length)]).flatten()

        # dft = np.zeros(output_length * 2)
        #
        # for i in range(start, start + output_length):
        #     idx = (i - start) * 2
        #
        #     for n in range(length):
        #         dft[idx] += series[n] * math.cos(2 * math.pi * n * i / length)
        #         dft[idx + 1] += -series[n] * math.sin(2 * math.pi * n * i / length)

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

        phis = np.array([[math.cos(2 * math.pi * (-((i * 2) + start_offset) / 2) / self.window_size),
                          -math.sin(2 * math.pi * (-((i * 2) + start_offset) / 2) / self.window_size)]
                         for i in range(0, int(length / 2))]).flatten()

        # phis = np.zeros(length)
        #
        # for i in range(0, length, 2):
        #     half = -(i + start_offset) / 2
        #     phis[i] = math.cos(2 * math.pi * half / self.window_size)
        #     phis[i + 1] = -math.sin(2 * math.pi * half / self.window_size)

        end = max(1, len(series) - self.window_size + 1)
        stds = self.calc_incremental_mean_std(series, end)
        transformed = np.zeros((end, length))
        mft_data = np.array([])

        for i in range(end):
            if i > 0:
                for n in range(0, length, 2):
                    real = mft_data[n] + series[i + self.window_size - 1] - series[i - 1]
                    imag = mft_data[n + 1]
                    mft_data[n] = real * phis[n] - imag * phis[n + 1]
                    mft_data[n + 1] = real * phis[n + 1] + phis[n] * imag
            else:
                mft_data = self.discrete_fourier_transform(series[0:self.window_size], normalise=False)

            normalising_factor = (1 / stds[i] if stds[i] > 0 else 1) * self.inverse_sqrt_win_size
            transformed[i] = mft_data * normalising_factor

        return transformed

    def calc_incremental_mean_std(self, series, end):
        means = np.zeros(end)
        stds = np.zeros(end)

        window = series[0:self.window_size]
        series_sum = np.sum(window)
        square_sum = np.sum(np.multiply(window, window))

        # series_sum = 0
        # square_sum = 0
        #
        # for ww in range(self.window_size):
        #     series_sum += series[ww]
        #     square_sum += series[ww] * series[ww]

        r_window_length = 1 / self.window_size
        means[0] = series_sum * r_window_length
        buf = square_sum * r_window_length - means[0] * means[0]
        stds[0] = math.sqrt(buf) if buf > 0 else 0

        for w in range(1, end):
            series_sum += series[w + self.window_size - 1] - series[w - 1]
            means[w] = series_sum * r_window_length
            square_sum += series[w + self.window_size - 1] * series[w + self.window_size - 1] - series[w - 1] * series[
                w - 1]
            buf = square_sum * r_window_length - means[w] * means[w]
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

        new_bags['dim_' + str(0)] = dim

        return new_bags

    def add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word.word == last_word:
            return last_word

        bag[word.word] = bag.get(word.word, 0) + 1

        # if word.word in bag:
        #     bag[word.word] += 1
        # else:
        #     bag[word.word] = 1

        return word.word
