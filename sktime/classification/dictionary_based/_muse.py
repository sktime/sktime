# -*- coding: utf-8 -*-
"""WEASEL+MUSE classifier.

multivariate dictionary based classifier based on SFA transform, dictionaries
and logistic regression.
"""

__author__ = ["patrickzib", "BINAYKUMAR943"]
__all__ = ["MUSE"]

import math
import warnings

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.panel.dictionary_based import SFA


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
    anova: boolean, default=True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given
    bigrams: boolean, default=True
        whether to create bigrams of SFA words
    window_inc: int, default=2
        WEASEL create a BoP model for each window sizes. This is the
        increment used to determine the next window size.
     p_threshold: int, default=0.05 (disabled by default)
        Feature selection is applied based on the chi-squared test.
        This is the p-value threshold to use for chi-squared test on bag-of-words
        (lower means more strict). 1 indicates that the test
        should not be performed.
    use_first_order_differences: boolean, default=True
        If set to True will add the first order differences of each dimension
        to the data.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state: int or None, default=None
        Seed for random, integer

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    WEASEL

    References
    ----------
    .. [1] Patrick Schäfer and Ulf Leser, "Multivariate time series classification
        with WEASEL+MUSE", in proc 3rd ECML/PKDD Workshop on AALTD}, 2018
        https://arxiv.org/abs/1711.11343

    Notes
    -----
    For the Java version, see
    `MUSE <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/multivariate/WEASEL_MUSE.java>`_.

    Examples
    --------
    >>> from sktime.classification.dictionary_based import MUSE
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = MUSE(window_inc=4, use_first_order_differences=False)
    >>> clf.fit(X_train, y_train)
    MUSE(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "X_inner_mtype": "nested_univ",  # MUSE requires nested dataframe
        "classifier_type": "dictionary",
    }

    def __init__(
        self,
        anova=True,
        bigrams=True,
        window_inc=2,
        p_threshold=0.05,
        use_first_order_differences=True,
        n_jobs=1,
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
        self.n_jobs = n_jobs

        super(MUSE, self).__init__()

    def _fit(self, X, y):
        """Build a WEASEL+MUSE classifiers from the training set (X, y).

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        y = np.asarray(y)

        # add first order differences in each dimension to TS
        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        # Window length parameter space dependent on series length
        self.col_names = X.columns

        rng = check_random_state(self.random_state)

        self.n_dims = len(self.col_names)
        self.highest_dim_bit = (math.ceil(math.log2(self.n_dims))) + 1

        if self.n_dims == 1:
            warnings.warn(
                "MUSE Warning: Input series is univariate; MUSE is designed for"
                + " multivariate series. It is recommended WEASEL is used instead."
            )

        def _parallel_fit(ind, column):
            all_words = [
                [] for x in range(X.shape[0])
            ]  # no dict needed, array is enough
            relevant_features_count = 0

            # On each dimension, perform SFA
            X_dim = X[[column]]
            X_dim = from_nested_to_3d_numpy(X_dim)
            series_length = X_dim.shape[-1]  # TODO compute minimum over all ts ?

            SFA_transformers = []

            # increment window size in steps of 'win_inc'
            win_inc = self._compute_window_inc(series_length)

            self.max_window = int(min(series_length, self.max_window))
            if self.min_window > self.max_window:
                raise ValueError(
                    f"Error in MUSE, min_window ="
                    f"{self.min_window} is bigger"
                    f" than max_window ={self.max_window}."
                    f" Try set min_window to be smaller than series length in "
                    f"the constructor, but the classifier may not work at "
                    f"all with very short series"
                )

            window_sizes = np.array(
                list(range(self.min_window, self.max_window, win_inc))
            )
            highest_bits = math.ceil(math.log2(self.max_window)) + 1

            for window_size in window_sizes:
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
                    n_jobs=self._threads_to_use,
                )

                sfa_words = transformer.fit_transform(X_dim, y)

                SFA_transformers.append(transformer)
                bag = sfa_words[0]
                apply_chi_squared = self.p_threshold < 1

                # chi-squared test to keep only relevant features
                relevant_features = {}
                if apply_chi_squared:
                    vectorizer = DictVectorizer(sparse=True, dtype=np.int32, sort=False)
                    bag_vec = vectorizer.fit_transform(bag)

                    chi2_statistics, p = chi2(bag_vec, y)
                    relevant_features_idx = np.where(p <= self.p_threshold)[0]
                    relevant_features = set(
                        np.array(vectorizer.feature_names_)[relevant_features_idx]
                    )
                    relevant_features_count += len(relevant_features_idx)

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(highest_bits)
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # chi-squared test
                        if (not apply_chi_squared) or (key in relevant_features):
                            # append the prefices to the words to
                            # distinguish between window-sizes
                            word = MUSE._shift_left(
                                key, highest, ind, self.highest_dim_bit, window_size
                            )
                            all_words[j].append((word, value))

            return (
                all_words,
                relevant_features_count,
                SFA_transformers,
                window_sizes,
                highest_bits,
            )

        parallel_res = Parallel(n_jobs=self._threads_to_use)(
            delayed(_parallel_fit)(ind, column)
            for (ind, column) in enumerate(self.col_names.values)
        )

        relevant_features_count = 0
        all_words = [dict() for x in range(len(X))]

        self.SFA_transformers = []
        self.window_sizes = []
        self.highest_bits = []

        for (
            sfa_words,
            rel_features_count,
            transformers,
            window_sizes,
            highest_bits,
        ) in parallel_res:
            relevant_features_count += rel_features_count

            self.window_sizes.append(window_sizes)
            self.SFA_transformers.append(transformers)
            self.highest_bits.append(highest_bits)

            for idx, bag in enumerate(sfa_words):
                all_words[idx].update(bag)

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
                n_jobs=self._threads_to_use,
            ),
        )

        for words in all_words:
            if len(words) == 0:
                words[-1] = 1

        self.clf.fit(all_words, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        bag = self._transform_words(X)
        return self.clf.predict(bag)

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        def _parallel_transform_words(X, ind, column):
            bag_all_words = [[] for _ in range(len(X))]

            # On each dimension, perform SFA
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
                        bag_all_words[j].append((word, value))
            return bag_all_words

        parallel_res = Parallel(n_jobs=self._threads_to_use)(
            delayed(_parallel_transform_words)(X, ind, column)
            for ind, column in enumerate(self.col_names)
        )
        all_words = [dict() for x in range(len(X))]
        for sfa_words in parallel_res:
            for idx, bag in enumerate(sfa_words):
                all_words[idx].update(bag)
        return all_words

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {"window_inc": 4, "use_first_order_differences": False}
