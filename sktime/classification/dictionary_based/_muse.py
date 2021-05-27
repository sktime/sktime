# -*- coding: utf-8 -*-
"""WEASEL+MUSE classifier.

multivariate dictionary based classifier based on SFA transform, dictionaries
and linear regression.
"""

__author__ = "Patrick Schäfer"
__all__ = ["MUSE"]

import math
import numpy as np
from numba import njit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import class_distribution

# from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFA
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class MUSE(BaseClassifier):
    """MUSE (MUltivariate Symbolic Extension).

    Also known as WEASLE-MUSE: implementation of multivariate version of WEASEL,
    referred to as just MUSE from [1].

    Overview: Input n series length m
     WEASEL+MUSE is a multivariate  dictionary classifier that builds a
     bag-of-patterns using SFA for different window lengths and learns a
     logistic regression classifier on this bag.

     There are these primary parameters:
             alphabet_size: alphabet size
             chi2-threshold: used for feature selection to select best words
             anova: select best l/2 fourier coefficients other than first ones
             bigrams: using bigrams of SFA words
             binning_strategy: the binning strategy used to disctrtize into
                               SFA words.


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

     p_threshold:      int, default = 0.05 (disabled by default)
        Feature selection is applied based on the chi-squared test.
        This is the p-value threshold to use for chi-squared test on bag-of-words
        (lower means more strict). 1 indicates that the test
        should not be performed.

    use_first_order_differences:    boolean, default = True
        If set to True will add the first order differences of each dimension
        to the data.

    random_state:        int or None,
        Seed for random, integer

    See Also
    --------
    WEASEL

    Notes
    -----
    ..[1] Patrick Schäfer and Ulf Leser, "Multivariate time series classification
    with WEASEL+MUSE",    in proc 3rd ECML/PKDD Workshop on AALTD}, 2018
    https://arxiv.org/abs/1711.11343
    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/multivariate/WEASEL_MUSE.java

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        anova=True,
        bigrams=True,
        window_inc=2,
        p_threshold=0.05,
        use_first_order_differences=True,
        random_state=None,
    ):

        # currently other values than 4 are not supported.
        self.alphabet_size = 4

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold
        self.anova = anova
        self.use_first_order_differences = use_first_order_differences

        self.norm_options = [False]
        self.word_lengths = [4, 6]

        self.bigrams = bigrams
        self.binning_strategies = ["equi-width", "equi-depth"]
        self.random_state = random_state

        self.min_window = 6
        self.max_window = 100

        self.window_inc = window_inc
        self.highest_bit = -1
        self.window_sizes = []

        self.col_names = []
        self.highest_dim_bit = 0
        self.highest_bits = []

        self.SFA_transformers = []
        self.clf = None
        self.classes_ = []

        super(MUSE, self).__init__()

    def fit(self, X, y):
        """Build a WEASEL+MUSE classifiers from the training set (X, y).

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, coerce_to_pandas=True)
        y = np.asarray(y)
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        # add first order differences in each dimension to TS
        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        # Window length parameter space dependent on series length
        self.col_names = X.columns

        rng = check_random_state(self.random_state)

        self.n_dims = len(self.col_names)
        self.highest_dim_bit = (math.ceil(math.log2(self.n_dims))) + 1
        self.highest_bits = np.zeros(self.n_dims)

        self.SFA_transformers = [[] for _ in range(self.n_dims)]

        # the words of all dimensions and all time series
        all_words = [dict() for _ in range(X.shape[0])]

        # On each dimension, perform SFA
        for ind, column in enumerate(self.col_names):
            X_dim = X[[column]]
            X_dim = from_nested_to_3d_numpy(X_dim)
            series_length = X_dim.shape[-1]  # TODO compute minimum over all ts ?

            # increment window size in steps of 'win_inc'
            win_inc = self._compute_window_inc(series_length)

            self.max_window = int(min(series_length, self.max_window))
            if self.min_window > self.max_window:
                raise ValueError(
                    f"Error in MUSE, min_window ="
                    f"{self.min_window} is bigger"
                    f" than max_window ={self.max_window},"
                    f" series length is {self.series_length}"
                    f" try set min_window to be smaller than series length in "
                    f"the constructor, but the classifier may not work at "
                    f"all with very short series"
                )
            self.window_sizes.append(
                list(range(self.min_window, self.max_window, win_inc))
            )

            self.highest_bits[ind] = math.ceil(math.log2(self.max_window)) + 1

            for window_size in self.window_sizes[ind]:

                transformer = SFA(
                    word_length=rng.choice(self.word_lengths),
                    alphabet_size=self.alphabet_size,
                    window_size=window_size,
                    norm=rng.choice(self.norm_options),
                    anova=self.anova,
                    binning_method=rng.choice(self.binning_strategies),
                    bigrams=self.bigrams,
                    remove_repeat_words=False,
                    lower_bounding=False,
                    save_words=False,
                )

                sfa_words = transformer.fit_transform(X_dim, y)

                self.SFA_transformers[ind].append(transformer)
                bag = sfa_words[0]

                # chi-squared test to keep only relevant features
                relevant_features = {}
                apply_chi_squared = self.p_threshold < 1
                if apply_chi_squared:
                    vectorizer = DictVectorizer(sparse=True, dtype=np.int32, sort=False)
                    bag_vec = vectorizer.fit_transform(bag)

                    chi2_statistics, p = chi2(bag_vec, y)
                    relevant_features_idx = np.where(p <= self.p_threshold)[0]
                    relevant_features = set(
                        np.array(vectorizer.feature_names_)[relevant_features_idx]
                    )

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(self.highest_bits[ind])
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # chi-squared test
                        if (not apply_chi_squared) or (key in relevant_features):
                            # append the prefices to the words to
                            # distinguish between window-sizes
                            word = MUSE._shift_left(
                                key, highest, ind, self.highest_dim_bit, window_size
                            )
                            all_words[j][word] = value

        self.clf = make_pipeline(
            DictVectorizer(sparse=True, sort=False),
            # StandardScaler(with_mean=True, copy=False),
            LogisticRegression(
                max_iter=5000,
                solver="liblinear",
                dual=True,
                # class_weight="balanced",
                penalty="l2",
                random_state=self.random_state,
            ),
        )

        self.clf.fit(all_words, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, 1]
        """
        bag = self._transform_words(X)
        return self.clf.predict(bag)

    def predict_proba(self, X):
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape [n, 1]

        Returns
        -------
        array of shape [n, self.n_classes]
        """
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_pandas=True)

        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        bag_all_words = [dict() for _ in range(len(X))]

        # On each dimension, perform SFA
        for ind, column in enumerate(self.col_names):
            X_dim = X[[column]]
            X_dim = from_nested_to_3d_numpy(X_dim)

            for i, window_size in enumerate(self.window_sizes[ind]):

                # SFA transform
                sfa_words = self.SFA_transformers[ind][i].transform(X_dim)
                bag = sfa_words[0]

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(self.highest_bits[ind])
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # append the prefices to the words to distinguish
                        # between window-sizes
                        word = MUSE._shift_left(
                            key, highest, ind, self.highest_dim_bit, window_size
                        )
                        bag_all_words[j][word] = value

        return bag_all_words

    def _add_first_order_differences(self, X):
        X_copy = X.copy()
        for column in X.columns:
            X_copy[str(column) + "_diff"] = X_copy[column]
            for ts in X[column]:
                ts_diff = ts.diff(1)
                ts.replace(ts_diff)
        return X_copy

    def _compute_window_inc(self, series_length):
        win_inc = self.window_inc
        if series_length < 100:
            win_inc = 1  # less than 100 is ok time-wise
        return win_inc

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _shift_left(key, highest, ind, highest_dim_bit, window_size):
        return ((key << highest | ind) << highest_dim_bit) | window_size
