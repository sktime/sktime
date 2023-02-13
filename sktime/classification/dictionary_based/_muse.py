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
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFAFast


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
    variance: boolean, default = False
            If True, the Fourier coefficient selection is done via the largest
            variance. If False, the first Fourier coefficients are selected.
            Only applicable if labels are given
    bigrams: boolean, default=True
        whether to create bigrams of SFA words
    window_inc: int, default=2
        WEASEL create a BoP model for each window sizes. This is the
        increment used to determine the next window size.
    alphabet_size : default = 4
        Number of possible letters (values) for each word.
    p_threshold: int, default=0.05 (disabled by default)
        Used when feature selection is applied based on the chi-squared test.
        This is the p-value threshold to use for chi-squared test on bag-of-words
        (lower means more strict). 1 indicates that the test
        should not be performed.
    use_first_order_differences: boolean, default=True
        If set to True will add the first order differences of each dimension
        to the data.
    support_probabilities: bool, default: False
        If set to False, a RidgeClassifierCV will be trained, which has higher accuracy
        and is faster, yet does not support predict_proba.
        If set to True, a LogisticRegression will be trained, which does support
        predict_proba(), yet is slower and typically less accuracy. predict_proba() is
        needed for example in Early-Classification like TEASER.
    feature_selection: {"chi2", "none", "random"}, default: chi2
        Sets the feature selections strategy to be used. Chi2 reduces the number
        of words significantly and is thus much faster (preferred). Random also reduces
        the number significantly. None applies not feature selectiona and yields large
        bag of words, e.g. much memory may be needed.
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
    - `Original Publication <https://github.com/patrickzib/SFA>`_.
    - `MUSE
        <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
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
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "classifier_type": "dictionary",
    }

    def __init__(
        self,
        anova=True,
        variance=False,
        bigrams=True,
        window_inc=2,
        alphabet_size=4,
        use_first_order_differences=True,
        feature_selection="chi2",
        p_threshold=0.05,
        support_probabilities=False,
        n_jobs=1,
        random_state=None,
    ):

        # currently other values than 4 are not supported.
        self.alphabet_size = alphabet_size

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold
        self.anova = anova
        self.variance = variance
        self.use_first_order_differences = use_first_order_differences

        self.norm_options = [False]
        self.word_lengths = [4, 6]

        self.bigrams = bigrams
        self.binning_strategies = ["equi-width", "equi-depth"]
        self.random_state = random_state

        self.min_window = 6
        self.max_window = 100

        self.window_inc = window_inc
        self.window_sizes = []

        self.SFA_transformers = []
        self.clf = None

        self.n_jobs = n_jobs
        self.support_probabilities = support_probabilities
        self.total_features_count = 0
        self.feature_selection = feature_selection

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
        self.n_dims = X.shape[1]

        self.highest_dim_bit = (math.ceil(math.log2(self.n_dims))) + 1

        if self.n_dims == 1:
            warnings.warn(
                "MUSE Warning: Input series is univariate; MUSE is designed for"
                + " multivariate series. It is recommended WEASEL is used instead."
            )

        if self.variance and self.anova:
            raise ValueError("MUSE Warning: Please set either variance or anova.")

        parallel_res = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_parallel_fit)(
                X,
                y.copy(),  # no clue why, but this copy is required.
                ind,
                self.min_window,
                self.max_window,
                self.window_inc,
                self.word_lengths,
                self.alphabet_size,
                self.norm_options,
                self.anova,
                self.variance,
                self.binning_strategies,
                self.bigrams,
                self.n_jobs,
                self.p_threshold,
                self.feature_selection,
                self.random_state,
            )
            for ind in range(self.n_dims)
        )

        self.SFA_transformers = [[] for _ in range(X.shape[1])]
        self.window_sizes = [[] for _ in range(X.shape[1])]
        all_words = []
        for (
            ind,
            sfa_words,
            transformer,
            window_sizes,
            rel_features_count,
        ) in parallel_res:
            self.SFA_transformers[ind].extend(transformer)
            self.window_sizes[ind].extend(window_sizes)
            all_words.extend(sfa_words)
            self.total_features_count += rel_features_count
        if type(all_words[0]) is np.ndarray:
            all_words = np.concatenate(all_words, axis=1)
        else:
            all_words = hstack((all_words))

        # Ridge Classifier does not give probabilities
        if not self.support_probabilities:
            self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        else:
            self.clf = LogisticRegression(
                max_iter=5000,
                solver="liblinear",
                dual=True,
                # class_weight="balanced",
                penalty="l2",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        self.clf.fit(all_words, y)
        self.total_features_count = all_words.shape[-1]
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
        if self.support_probabilities:
            return self.clf.predict_proba(bag)
        else:
            raise ValueError(
                "Error in MUSE, please set support_probabilities=True, to"
                + "allow for probabilities to be computed."
            )

    def _transform_words(self, X):
        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        parallel_res = Parallel(n_jobs=self._threads_to_use, backend="threading")(
            delayed(_parallel_transform_words)(
                X, self.window_sizes, self.SFA_transformers, ind
            )
            for ind in range(self.n_dims)
        )

        all_words = []
        for sfa_words in parallel_res:
            all_words.extend(sfa_words)
        if type(all_words[0]) is np.ndarray:
            all_words = np.concatenate(all_words, axis=1)
        else:
            all_words = hstack((all_words))

        return all_words

    def _add_first_order_differences(self, X):
        X_new = np.zeros((X.shape[0], X.shape[1] * 2, X.shape[2]))
        X_new[:, 0 : X.shape[1], :] = X
        diff = np.diff(X, 1)
        X_new[:, X.shape[1] :, : diff.shape[2]] = diff
        return X_new

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
        return {
            "window_inc": 4,
            "alphabet_size": 2,
            "use_first_order_differences": False,
            "support_probabilities": True,
            "feature_selection": "none",
            "bigrams": False,
        }


def _compute_window_inc(series_length, window_inc):
    win_inc = window_inc
    if series_length < 100:
        win_inc = 1  # less than 100 is ok time-wise

    return win_inc


def _parallel_transform_words(X, window_sizes, SFA_transformers, ind):
    # On each dimension, perform SFA
    X_dim = X[:, ind]

    bag_all_words = []
    for i in range(len(window_sizes[ind])):
        words = SFA_transformers[ind][i].transform(X_dim)
        bag_all_words.append(words)

    return bag_all_words


def _parallel_fit(
    X,
    y,
    ind,
    min_window,
    max_window,
    window_inc,
    word_lengths,
    alphabet_size,
    norm_options,
    anova,
    variance,
    binning_strategies,
    bigrams,
    n_jobs,
    p_threshold,
    feature_selection,
    random_state,
):
    if random_state is not None:
        rng = check_random_state(random_state + ind)
    else:
        rng = check_random_state(random_state)

    all_words = []

    # On each dimension, perform SFA
    X_dim = X[:, ind]
    series_length = X_dim.shape[-1]

    # increment window size in steps of 'win_inc'
    win_inc = _compute_window_inc(series_length, window_inc)
    max_window = int(min(series_length, max_window))

    if min_window > max_window:
        raise ValueError(
            f"Error in MUSE, min_window ="
            f"{min_window} is bigger"
            f" than max_window ={max_window}."
            f" Try set min_window to be smaller than series length in "
            f"the constructor, but the classifier may not work at "
            f"all with very short series"
        )

    SFA_transformers = []
    window_sizes = np.arange(min_window, max_window, win_inc)
    relevant_features_count = 0

    for window_size in window_sizes:
        transformer = SFAFast(
            word_length=rng.choice(word_lengths),
            alphabet_size=alphabet_size,
            window_size=window_size,
            norm=rng.choice(norm_options),
            anova=anova,
            variance=variance,
            binning_method=rng.choice(binning_strategies),
            bigrams=bigrams,
            remove_repeat_words=False,
            lower_bounding=False,
            p_threshold=p_threshold,
            feature_selection=feature_selection,
            save_words=False,
            n_jobs=n_jobs,
            return_pandas_data_series=False,
            return_sparse=True,
        )

        all_words.append(transformer.fit_transform(X_dim, y))
        SFA_transformers.append(transformer)
        relevant_features_count = transformer.feature_count

    return ind, all_words, SFA_transformers, window_sizes, relevant_features_count
