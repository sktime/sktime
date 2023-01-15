# -*- coding: utf-8 -*-
"""Symbolic Aggregate approXimation (SAX) transformer."""

import sys

import numpy as np
import pandas as pd
import scipy.stats

from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.dictionary_based import PAA

#    TO DO: verify this returned pandas is consistent with sktime
#    definition. Timestamps?

# from numba import types
# from numba.experimental import jitclass

__author__ = "MatthewMiddlehurst"


class SAX(BaseTransformer):
    """Symbolic Aggregate approXimation (SAX) transformer.

    as described in
    Jessica Lin, Eamonn Keogh, Li Wei and Stefano Lonardi,
    "Experiencing SAX: a novel symbolic representation of time series"
    Data Mining and Knowledge Discovery, 15(2):107-144
    Overview: for each series:
        run a sliding window across the series
        for each window
            shorten the series with PAA (Piecewise Approximate Aggregation)
            discretise the shortened series into fixed bins
            form a word from these discrete values
    by default SAX produces a single word per series (window_size=0).
    SAX returns a pandas data frame where column 0 is the histogram (sparse
    pd.series)
    of each series.

    Parameters
    ----------
    word_length:         int, length of word to shorten window to (using
    PAA) (default 8)
    alphabet_size:       int, number of values to discretise each value
    to (default to 4)
    window_size:         int, size of window for sliding. Input series
    length for whole series transform (default to 12)
    remove_repeat_words: boolean, whether to use numerosity reduction (
    default False)
    save_words:          boolean, whether to use numerosity reduction (
    default False)

    return_pandas_data_series:          boolean, default = True
        set to true to return Pandas Series as a result of transform.
        setting to true reduces speed significantly but is required for
        automatic test.

    Attributes
    ----------
    words:      history = []
    """

    _tags = {
        "univariate-only": True,
        "fit_is_empty": True,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict require for y?
    }

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        remove_repeat_words=False,
        save_words=False,
        return_pandas_data_series=True,
    ):
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.return_pandas_data_series = return_pandas_data_series
        self.words = []

        super(SAX, self).__init__(_output_convert="off")

    # todo: looks like this just loops over series instances
    # so should be refactored to work on Series directly
    def _transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """
        X = X.squeeze(1)

        if self.alphabet_size < 2 or self.alphabet_size > 4:
            raise RuntimeError("Alphabet size must be an integer between 2 and 4")
        if self.word_length < 1 or self.word_length > 16:
            raise RuntimeError("Word length must be an integer between 1 and 16")

        breakpoints = self._generate_breakpoints()
        n_instances, series_length = X.shape

        bags = pd.DataFrame()
        dim = []

        for i in range(n_instances):
            bag = {}
            lastWord = -1

            words = []

            num_windows_per_inst = series_length - self.window_size + 1
            split = np.array(
                X[
                    i,
                    np.arange(self.window_size)[None, :]
                    + np.arange(num_windows_per_inst)[:, None],
                ]
            )

            split = scipy.stats.zscore(split, axis=1)

            paa = PAA(num_intervals=self.word_length)
            data = pd.DataFrame()
            data[0] = [pd.Series(x, dtype=np.float32) for x in split]
            patterns = paa.fit_transform(data)
            patterns = np.asarray([a.values for a in patterns.iloc[:, 0]])

            for n in range(patterns.shape[0]):
                pattern = patterns[n, :]
                word = self._create_word(pattern, breakpoints)
                words.append(word)
                lastWord = self._add_to_bag(bag, word, lastWord)

            if self.save_words:
                self.words.append(words)

            dim.append(pd.Series(bag) if self.return_pandas_data_series else bag)

        bags[0] = dim

        return bags

    def _create_word(self, pattern, breakpoints):
        word = 0
        for i in range(self.word_length):
            for bp in range(self.alphabet_size):
                if pattern[i] <= breakpoints[bp]:
                    word = (word << 2) | bp
                    break

        return word

    def _add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word == last_word:
            return False
        bag[word] = bag.get(word, 0) + 1
        return True

    def _generate_breakpoints(self):
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
            10: [
                -1.28,
                -0.84,
                -0.52,
                -0.25,
                0.0,
                0.25,
                0.52,
                0.84,
                1.28,
                sys.float_info.max,
            ],
        }[self.alphabet_size]

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
        # small word length, window size for testing
        params = {"word_length": 2, "window_size": 4}
        return params
