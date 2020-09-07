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
from sklearn.feature_selection import chi2
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

    # TODO
    # Overview: Input n series length m
    # BOSS performs a gird search over a set of parameter values, evaluating
    # each with a LOOCV. It then retains
    # all ensemble members within 92% of the best. There are three primary
    # parameters:
    #         alpha: alphabet size
    #         w: window length
    #         l: word length.
    # for any combination, a single BOSS slides a window length w along the
    # series. The w length window is shortened to
    # an l length word through taking a Fourier transform and keeping the
    # first l/2 complex coefficients. These l
    # coefficents are then discretised into alpha possible values, to form a
    # word length l. A histogram of words for each
    # series is formed and stored. fit involves finding n histograms.
    #
    # predict uses 1 nearest neighbour with a bespoke distance function.

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/dictionary_based/WEASEL.java


    Parameters
    ----------
    ensemble members rather than cross validate (cBOSS) (default=False)

    Attributes
    ----------


    """

    def __init__(self,
                 alphabet_size=4,
                 chi2_threshold=2,
                 anova=True,
                 bigrams=True,
                 binning_strategy="information-gain",
                 random_state=None
                 ):

        self.alphabet_size = alphabet_size
        self.anova = anova

        self.norm_options = [True, False]
        self.word_lengths = [4, 6]  # 6

        self.bigrams = bigrams
        self.chi2_threshold = chi2_threshold
        self.binning_strategy = binning_strategy
        self.random_state = random_state

        self.min_window = 6
        self.max_window = 350
        self.win_inc = 1
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

            # transformers = [SFA() for _ in range(len(self.window_sizes))]
            # for i, window_size in enumerate(self.window_sizes):
            #    transformers[i] =\
            #        SFA(word_length=np.max(self.word_lengths),
            #            alphabet_size=self.alphabet_size,
            #            window_size=window_size,
            #            norm=norm,
            #            anova=self.anova,
            #            binning_method=self.binning_strategy,
            #            bigrams=self.bigrams,
            #            remove_repeat_words=False,
            #            lower_bounding=False,
            #            save_words=True)
            #    transformers[i].fit_transform(X, y)

            for word_length in self.word_lengths:  # use the shortening trick??
                X_all_words = [dict() for x in range(len(X))]
                transformers = [SFA() for _ in range(len(self.window_sizes))]

                for i, window_size in enumerate(self.window_sizes):
                    # X_sfas = transformers[i]._shorten_bags(
                    #     word_length, np.max(self.word_lengths))

                    transformers[i] = SFA(word_length=word_length,
                                          alphabet_size=self.alphabet_size,
                                          window_size=window_size,
                                          norm=norm,
                                          anova=self.anova,
                                          binning_method=self.binning_strategy,
                                          bigrams=self.bigrams,
                                          remove_repeat_words=False,
                                          lower_bounding=False,
                                          save_words=False,
                                          skip_series_conversion=True)
                    X_sfas = transformers[i].fit_transform(X, y)

                    # TODO refactor: dicts not really needed here ...
                    # bag = [series.to_dict() for series in X_sfas.iloc[:, 0]]
                    bag = X_sfas.iloc[:, 0]

                    # chi-squared test to keep only relevent features
                    # bag_vec = DictVectorizer(sparse=False).fit_transform(bag)
                    # chi2_statistics, p = chi2(bag_vec, y)
                    # relevant_features = np.where(
                    #     chi2_statistics >= self.chi2_threshold)[0]

                    # merging bag-of-patterns of different window_sizes
                    # to single bag-of-patterns with prefix indicating
                    # the used window-length
                    for j in range(len(bag)):
                        for (key, value) in bag[j].items():
                            # if key in relevant_features:  # chi-squared test
                            # append the prefices to the words to
                            # distinguish between window-sizes
                            word = (key << self.highest_bit) | window_size
                            X_all_words[j][word] = value  # add value of 1?

                # TODO use CountVectorizer instead on actual words ... ???
                vectorizer = DictVectorizer(sparse=True)
                bag_vec = vectorizer.fit_transform(X_all_words)
                lr = LogisticRegression(max_iter=5000, solver="liblinear",
                                        dual=True, n_jobs=-1, penalty="l2",
                                        random_state=self.random_state)

                # cross validation scores
                current_acc = cross_val_score(lr, bag_vec, y, cv=5).mean()
                print("Train acc:", norm, current_acc)

                if current_acc > max_acc:
                    max_acc = current_acc
                    self.vectorizer = vectorizer
                    self.clf = lr
                    self.SFA_transformers = transformers
                    self.best_word_length = word_length
                    final_bag_vec = bag_vec

                if max_acc == 1.0:
                    break

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
            X_sfa = self.SFA_transformers[i].transform(X)
            X_words = X_sfa.iloc[:, 0]

            # merging bag-of-patterns of different window_sizes
            # to single bag-of-patterns with prefix indicating
            # the used window-length
            for j in range(len(X_words)):
                for (key, value) in X_words[j].items():
                    # append the prefices to the words to distinguish
                    # between window-sizes
                    word = (key << self.highest_bit) | window_size
                    bag_all_words[j][word] = value

        return bag_all_words
