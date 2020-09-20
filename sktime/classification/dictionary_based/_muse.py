""" WEASEL+MUSE classifier
multivariate dictionary based classifier based on SFA transform, dictionaries
and linear regression.
"""

__author__ = "Patrick Schäfer"
__all__ = ["MUSE"]

import math
import numpy as np
import pandas as pd

from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y
from sktime.utils.data_container import tabularize

from sklearn.utils import check_random_state
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from numba import njit
# from numba.typed import Dict


class MUSE(BaseClassifier):
    """ WEASEL+MUSE (MUltivariate Symbolic Extension)

    MUSE: implementation of MUSE from Schäfer:
    @article{schafer2017multivariate,
      title={Multivariate time series classification with WEASEL+MUSE},
      author={Sch{\"a}fer, Patrick and Leser, Ulf},
      journal={3rd ECML/PKDD Workshop on AALTD},
      year={2018}
    }

    # Overview: Input n series length m
    # WEASEL+MUSE is a multivariate  dictionary classifier that builds a
    # bag-of-patterns using SFA for different window lengths and learns a
    # logistic regression classifier on this bag.
    #
    # There are these primary parameters:
    #         alphabet_size: alphabet size
    #         chi2-threshold: used for feature selection to select best words
    #         anova: select best l/2 fourier coefficients other than first ones
    #         bigrams: using bigrams of SFA words
    #         binning_strategy: the binning strategy used to disctrtize into
    #                           SFA words.
    #


    Parameters
    ----------
    anova:               boolean, default = True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given

    bigrams:             boolean, default = True
        whether to create bigrams of SFA words

    window_inc:          int, default = 4
        WEASEL create a BoP model for each window sizes. This is the
        increment used to determine the next window size.

    chi2_threshold:      int, default = 2 (enabled by default)
        Feature selection is applied based on the chi-squared test.
        This is the threshold to use for chi-squared test on bag-of-words
        (higher means more strict). Negative values indicate that the test
        should not be performed.

    random_state:        int or None,
        Seed for random, integer

    Attributes
    ----------


    """

    def __init__(self,
                 anova=True,
                 bigrams=True,
                 window_inc=4,
                 chi2_threshold=2,
                 random_state=None
                 ):

        # currently other values than 4 are not supported.
        self.alphabet_size = 4

        # feature selection is applied based on the chi-squared test.
        self.chi2_threshold = chi2_threshold

        self.anova = anova

        self.norm_options = [True, False]
        self.word_lengths = [4]  # 6

        self.bigrams = bigrams
        self.binning_strategies = ["equi-width", "equi-depth"]
        self.random_state = random_state

        self.min_window = 4
        self.max_window = 350

        self.window_inc = window_inc
        self.highest_bit = -1
        self.window_sizes = []

        self.col_names = []
        self.highest_dim_bit = 0
        self.highest_bits = []

        self.SFA_transformers = []
        self.clf = None
        self.best_word_length = -1

        super(MUSE, self).__init__()

    def fit(self, X, y):
        """Build a WEASEL+MUSE classifiers from the training set (X, y),

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y, enforce_univariate=False)
        y = y.values if isinstance(y, pd.Series) else y

        # TODO we currently only support 2^5 dimensions

        # Window length parameter space dependent on series length
        self.col_names = X.columns

        rng = check_random_state(self.random_state)

        self.n_dims = len(self.col_names)
        self.highest_dim_bit = (math.ceil(math.log2(self.n_dims))) + 1
        self.highest_bits = np.zeros(self.n_dims)

        self.SFA_transformers = [[] for _ in range(self.n_dims)]

        # the words of all dimensions and all time series
        all_words = [dict() for _ in range(X.shape[0])]

        # TODO add first order differences in each dimension to TS

        # On each dimension, perform SFA
        for ind, column in enumerate(self.col_names):
            X_dim = X[column]
            X_dim = tabularize(X_dim, return_array=True)
            series_length = len(X_dim[0])  # TODO compute minimum over all ts ?

            # increment window size in steps of 'win_inc'
            win_inc = self.compute_window_inc(series_length)

            self.max_window = int(min(series_length, self.max_window))
            self.window_sizes.append(list(range(self.min_window,
                                                self.max_window,
                                                win_inc)))

            self.highest_bits[ind] = (math.ceil(math.log2(self.max_window))) + 1

            for i, window_size in enumerate(self.window_sizes[ind]):

                transformer = SFA(word_length=rng.choice(self.word_lengths),
                                  alphabet_size=self.alphabet_size,
                                  window_size=window_size,
                                  norm=rng.choice(self.norm_options),
                                  anova=self.anova,
                                  binning_method=
                                  rng.choice(self.binning_strategies),
                                  bigrams=self.bigrams,
                                  remove_repeat_words=False,
                                  lower_bounding=False,
                                  save_words=False)

                sfa_words = transformer.fit_transform(X_dim, y)

                self.SFA_transformers[ind].append(transformer)
                bag = sfa_words[0]  # .iloc[:, 0]

                # chi-squared test to keep only relevant features
                relevant_features = {}
                apply_chi_squared = self.chi2_threshold > 0
                if apply_chi_squared:
                    bag_vec \
                        = DictVectorizer(sparse=False).fit_transform(bag)
                    chi2_statistics, p = chi2(bag_vec, y)
                    relevant_features = np.where(
                       chi2_statistics >= self.chi2_threshold)[0]

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(self.highest_bits[ind])
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # chi-squared test
                        if (not apply_chi_squared) or \
                                (key in relevant_features):
                            # append the prefices to the words to
                            # distinguish between window-sizes
                            word = MUSE.shift_left(key, highest, ind,
                                                   self.highest_dim_bit,
                                                   window_size)

                            all_words[j][word] = value

        self.clf = make_pipeline(
            DictVectorizer(sparse=False),
            StandardScaler(with_mean=True, copy=False),
            LogisticRegression(max_iter=5000,
                               solver="liblinear",
                               dual=True,
                               # class_weight="balanced",
                               penalty="l2",
                               random_state=self.random_state)
            )

        self.clf.fit(all_words, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        bag = self._transform_words(X)
        return self.clf.predict(bag)

    def predict_proba(self, X):
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False)
        bag_all_words = [dict() for _ in range(len(X))]

        # On each dimension, perform SFA
        for ind, column in enumerate(self.col_names):
            X_dim = X[column]
            X_dim = tabularize(X_dim, return_array=True)

            for i, window_size in enumerate(self.window_sizes[ind]):

                # SFA transform
                sfa_words = self.SFA_transformers[ind][i].transform(X_dim)
                bag = sfa_words[0]  # .iloc[:, 0]

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(self.highest_bits[ind])
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # append the prefices to the words to distinguish
                        # between window-sizes
                        word = MUSE.shift_left(key, highest, ind,
                                               self.highest_dim_bit,
                                               window_size)

                        bag_all_words[j][word] = value

        return bag_all_words

    def compute_window_inc(self, series_length):
        win_inc = self.window_inc
        if series_length < 50:
            win_inc = 2  # less than 50 is ok time-wise
        elif series_length < 100:
            win_inc = min(self.window_inc, 4)  # less than 50 is ok time-wise
        return win_inc

    @staticmethod
    @njit(fastmath=True, cache=True)
    def shift_left(key, highest, ind, highest_dim_bit, window_size):
        return ((key << highest | ind) << highest_dim_bit) | window_size
