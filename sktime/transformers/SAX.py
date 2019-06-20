import sys

import numpy as np
import pandas as pd
import sktime.transformers.shapelets as shapelets

from sktime.utils.bitword import BitWord
from sktime.transformers.base import BaseTransformer


class SAX(BaseTransformer):

    def __init__(self,
                 word_length,
                 alphabet_size,
                 window_size=0,
                 remove_repeat_words=False,
                 dim_to_use=0,
                 save_words=False
                 ):
        self.words = []
        self.breakpoints = []

        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size

        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words

        self.dim_to_use = dim_to_use

        self.num_insts = 0
        self.num_atts = 0

    def fit(self, X, **kwargs):
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

        self.num_atts = X.shape[1]

        if self.window_size == 0:
            self.window_size = self.num_atts

        self.breakpoints = self.generate_breakpoints()

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
            bag = {}
            lastWord = None

            words = []

            num_windows_per_inst = self.num_atts - self.window_size + 1
            split = np.array(X[i, np.arange(self.window_size)[None, :] + np.arange(num_windows_per_inst)[:, None]])

            for window in range(split.shape[0]):
                pattern = shapelets.RandomShapeletTransform.zscore(split[window])  # lazy code
                pattern = self.PAA(pattern)
                word = self.create_word(pattern)
                words.append(word)
                lastWord = self.add_to_bag(bag, word, lastWord)

            if self.save_words:
                self.words.append(words)

            dim.append(pd.Series(bag))

        bags['dim_' + str(self.dim_to_use)] = dim

        return bags

    def PAA(self, series):
        frames = []
        current_frame = 0
        current_frame_size = 0
        frame_length = self.num_atts / self.word_length
        frame_sum = 0

        for i in range(self.num_atts):
            remaining = frame_length - current_frame_size

            if remaining > 1:
                frame_sum += series[i]
                current_frame_size += 1
            else:
                frame_sum += remaining * series[i]
                current_frame_size += remaining

            if current_frame_size == frame_length:
                frames.append(frame_sum / frame_length)
                current_frame += 1

                frame_sum = (1-remaining) * series[i]
                current_frame_size = (1-remaining)

        # if the last frame was lost due to double imprecision
        if current_frame == self.word_length-1:
            frames.append(frame_sum / frame_length)

        return frames

    def create_word(self, dft):
        word = BitWord()

        for i in range(self.word_length):
            for bp in range(self.alphabet_size):
                if dft[i] <= self.breakpoints[i][bp]:
                    word.push(bp)
                    break

        return word

    def add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word.word == last_word:
            return last_word

        if word.word in bag:
            bag[word.word] += 1
        else:
            bag[word.word] = 1

        return word.word

    def generate_breakpoints(self):
        # Pre-made gaussian curve breakpoints from UEA TSC codebase
        return {
            2: [0, sys.float_info.max],
            3: [-0.43, 0.43, sys.float_info.max],
            4: [-0.67, 0, 0.67, sys.float_info.max],
            5: [-0.84, -0.25, 0.25, 0.84, sys.float_info.max],
            6: [-0.97, -0.43, 0, 0.43, 0.97, sys.float_info.max],
            7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, sys.float_info.max],
            8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, sys.float_info.max],
            9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, sys.float_info.max],
            10: [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28, sys.float_info.max]
        }[self.alphabet_size]
