""" WEASEL classifier
dictionary based classifier based on SFA transform, BOSS and linear regression.
"""

__author__ = "Patrick Schäfer"
__all__ = ["WEASEL"]

import math

import numpy as np
import pandas as pd
from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score

# from sktime.transformers.series_as_features.dictionary_based._sax import \
#     _BitWord

# from numba import njit
# from numba.typed import Dict


class WEASEL(BaseClassifier):
    """ Word ExtrAction for time SEries cLassification (WEASEL)

    WEASEL: implementation of WEASEL from Schäfer:
    @inproceedings{schafer2017fast,
      title={Fast and Accurate Time Series Classification with WEASEL},
      author={Sch{\"a}fer, Patrick and Leser, Ulf},
      booktitle={Proceedings of the 2017 ACM on Conference on Information and
                 Knowledge Management},
      pages={637--646},
      year={2017}
    }
    # Overview: Input n series length m
    # WEASEL is a dictionary classifier that builds a bag-of-patterns using SFA
    # for different window lengths and learns a logistic regression classifier
    # on this bag.
    #
    # There are these primary parameters:
    #         alphabet_size: alphabet size
    #         chi2-threshold: used for feature selection to select best words
    #         anova: select best l/2 fourier coefficients other than first ones
    #         bigrams: using bigrams of SFA words
    #         binning_strategy: the binning strategy used to discretise into
    #                           SFA words.
    #
    # WEASEL slides a window length w along the series. The w length window
    # is shortened to an l length word through taking a Fourier transform and
    # keeping the best l/2 complex coefficients using an anova one-sided
    # test. These l coefficients are then discretised into alpha possible
    # symbols, to form a word of length l. A histogram of words for each
    # series is formed and stored.
    # For each window-length a bag is created and all words are joint into
    # one bag-of-patterns. Words from different window-lengths are
    # discriminated by different prefixes.
    #
    # fit involves training a logistic regression classifier on the single
    # bag-of-patterns.
    #
    # predict uses the logistic regression classifier

    # For the Java version, see
    # https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    # /tsml/classifiers/dictionary_based/WEASEL.java


    Parameters
    ----------
    anova:               boolean, default = True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given

    bigrams:             boolean, default = True
            whether to create bigrams of SFA words

    binning_strategy:   {"equi-depth", "equi-width", "information-gain"},
                        default="information-gain"
        The binning method used to derive the breakpoints.

    random_state:       int or None,
        Seed for random, integer

    Attributes
    ----------


    """

    def __init__(self,
                 anova=True,
                 bigrams=True,
                 binning_strategy="information-gain",
                 random_state=None
                 ):

        # currently other values than 4 are not supported.
        self.alphabet_size = 4

        # feature selection is applied based on the chi-squared test.
        # this is the threshold to use for chi-squared test on bag-of-words
        # (higher means more strict)
        self.chi2_threshold = 2,

        self.anova = anova

        self.norm_options = [True, False]
        self.word_lengths = [4, 6]

        self.bigrams = bigrams
        self.binning_strategy = binning_strategy
        self.random_state = random_state

        self.min_window = 6
        self.max_window = 350

        # differs from publication. here set to 4 for performance reasons
        self.win_inc = 4
        self.highest_bit = -1
        self.window_sizes = []

        self.series_length = 0
        self.n_instances = 0

        self.SFA_transformers = []
        self.clf = None
        self.vectorizer = None
        self.best_word_length = -1

        super(WEASEL, self).__init__()

    def fit(self, X, y):
        """Build a WEASEL classifiers from the training set (X, y),

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y, enforce_univariate=True)
        y = y.values if isinstance(y, pd.Series) else y

        # Window length parameter space dependent on series length
        self.n_instances, self.series_length = X.shape[0], len(X.iloc[0, 0])
        self.max_window = min(self.series_length, self.max_window)
        self.window_sizes = list(range(self.min_window,
                                       self.max_window,
                                       self.win_inc))

        max_acc = -1
        self.highest_bit = (math.ceil(math.log2(self.max_window))+1)

        final_bag_vec = None

        for norm in self.norm_options:
            # transformers = []

            for w, word_length in enumerate(self.word_lengths):
                all_words = [dict() for x in range(len(X))]
                transformers = []

                for i, window_size in enumerate(self.window_sizes):
                    # if w == 0:  # only compute once, otherwise shorten
                    transformer = SFA(word_length=np.max(word_length),
                                      alphabet_size=self.alphabet_size,
                                      window_size=window_size,
                                      norm=norm,
                                      anova=self.anova,
                                      binning_method=self.binning_strategy,
                                      bigrams=self.bigrams,
                                      remove_repeat_words=False,
                                      lower_bounding=False,
                                      save_words=False)
                    sfa_words = transformer.fit_transform(X, y)
                    transformers.append(transformer)

                    # use the shortening of words trick
                    # sfa_words = transformers[i]._shorten_bags(word_length)

                    # TODO refactor? dicts not really needed here ...
                    bag = sfa_words.iloc[:, 0]

                    # chi-squared test to keep only relevent features
                    # bag_vec = DictVectorizer(sparse=False).fit_transform(bag)
                    # chi2_statistics, p = chi2(bag_vec, y)
                    # relevant_features = np.where(
                    #    chi2_statistics >= self.chi2_threshold)[0]

                    # merging bag-of-patterns of different window_sizes
                    # to single bag-of-patterns with prefix indicating
                    # the used window-length
                    for j in range(len(bag)):
                        for (key, value) in bag[j].items():
                            # if key in relevant_features:  # chi-squared test
                            # append the prefices to the words to
                            # distinguish between window-sizes
                            word = (key << self.highest_bit) | window_size
                            # X_all_words[j].append((word, value))
                            all_words[j][word] = value

                # TODO use CountVectorizer instead on actual words ... ???
                vectorizer = DictVectorizer(sparse=True)
                bag_vec = vectorizer.fit_transform(all_words)

                clf = LogisticRegression(max_iter=5000, solver="liblinear",
                                         dual=True, penalty="l2",
                                         random_state=self.random_state)
                current_acc = cross_val_score(clf, bag_vec, y, cv=5).mean()

                # clf = RandomForestClassifier(oob_score=True,
                #                              n_estimators=1000,
                #                              n_jobs=-1).fit(bag_vec, y)
                # current_acc = clf.oob_score_

                # print("Train acc:", norm, word_length, current_acc)

                if current_acc > max_acc:
                    max_acc = current_acc
                    self.vectorizer = vectorizer
                    self.clf = clf
                    self.SFA_transformers = transformers
                    self.best_word_length = word_length
                    final_bag_vec = bag_vec

                if max_acc == 1.0:
                    break  # there can be no better model than 1.0

        # # fit final model using all words
        # for i, window_size in enumerate(self.window_sizes):
        #     self.SFA_transformers[i] = \
        #         SFA(word_length=np.max(self.word_lengths),
        #             alphabet_size=self.alphabet_size,
        #             window_size=window_size,
        #             norm=norm,
        #             anova=self.anova,
        #             binning_method=self.binning_strategy,
        #             bigrams=self.bigrams,
        #             remove_repeat_words=False,
        #             lower_bounding=False,
        #             save_words=False)
        #     self.SFA_transformers[i].fit_transform(X, y)

        self.clf.fit(final_bag_vec, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        bag = self._transform_words(X)
        bag_dict = self.vectorizer.transform(bag)
        return self.clf.predict(bag_dict)

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        bag = self._transform_words(X)
        bag_dict = self.vectorizer.transform(bag)
        return self.clf.predict_proba(bag_dict)

    def _transform_words(self, X):
        bag_all_words = [dict() for _ in range(len(X))]
        for i, window_size in enumerate(self.window_sizes):

            # SFA transform
            sfa_words = self.SFA_transformers[i].transform(X)
            bag = sfa_words.iloc[:, 0]

            # merging bag-of-patterns of different window_sizes
            # to single bag-of-patterns with prefix indicating
            # the used window-length
            for j in range(len(bag)):
                for (key, value) in bag[j].items():
                    # append the prefices to the words to distinguish
                    # between window-sizes
                    word = (key << self.highest_bit) | window_size
                    bag_all_words[j][word] = value

        return bag_all_words
