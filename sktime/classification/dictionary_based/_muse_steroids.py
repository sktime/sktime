# -*- coding: utf-8 -*-
"""WEASEL+MUSE classifier.

multivariate dictionary based classifier based on SFA transform, dictionaries
and logistic regression.
"""

__author__ = ["patrickzib"]
__all__ = ["MUSE_STEROIDS"]

import math
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFAFast


class MUSE_STEROIDS(BaseClassifier):
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
    `MUSE <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/multivariate/WEASEL_MUSE.java>`_.

    Examples
    --------
    >>> from sktime.classification.dictionary_based import MUSE
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = MUSE_STEROIDS(window_inc=4, use_first_order_differences=False)
    >>> clf.fit(X_train, y_train)
    MUSE_STEROIDS(...)
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
        anova=False,
        variance=True,
        bigrams=False,
        binning_strategies=["equi-depth", "equi-width"],
        ensemble_size=50,
        max_feature_count=20_000,
        min_window=4,
        max_window=24,
        norm_options=[False],
        word_lengths=[8],
        alphabet_sizes=[2],
        use_first_differences=False,
        feature_selection="chi2",
        support_probabilities=False,
        n_jobs=1,
        random_state=None,
    ):

        self.alphabet_sizes = alphabet_sizes

        # feature selection is applied based on the chi-squared test.
        self.anova = anova
        self.variance = variance
        self.use_first_differences = use_first_differences

        self.norm_options = norm_options
        self.word_lengths = word_lengths
        self.min_window = min_window
        self.max_window = max_window
        self.ensemble_size = ensemble_size
        self.max_feature_count = max_feature_count
        self.feature_selection = feature_selection
        self.binning_strategies = binning_strategies

        self.bigrams = bigrams
        self.random_state = random_state

        self.window_sizes = []
        self.SFA_transformers = []
        self.clf = None

        self.n_jobs = n_jobs
        self.support_probabilities = support_probabilities
        self.total_features_count = 0
        self.feature_selection = feature_selection

        super(MUSE_STEROIDS, self).__init__()

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
        if self.use_first_differences:
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

        self.series_length = X.shape[-1]
        self.max_window = int(min(self.series_length, self.max_window))
        if self.min_window > self.max_window:
            raise ValueError(
                f"Error in WEASEL, min_window ="
                f"{self.min_window} is bigger"
                f" than max_window ={self.max_window},"
                f" series length is {self.series_length}"
                f" try set min_window to be smaller than series length in "
                f"the constructor, but the classifier may not work at "
                f"all with very short series"
            )

        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_parallel_fit)(
                ind,
                X,
                y.copy(),  # no clue why, but this copy is required.
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.feature_selection,
                self.random_state,
            )
            # for ind in range(self.n_dims)
            for ind in range(self.ensemble_size)
        )

        self.SFA_transformers = []
        all_words = []
        for (
            sfa_words,
            transformer,
        ) in parallel_res:
            self.SFA_transformers.append(transformer)  # Append! Not Extent! 2d Array
            all_words.extend(sfa_words)

        if type(all_words[0]) is np.ndarray:
            all_words = np.concatenate(all_words, axis=1)
        else:
            all_words = hstack((all_words))

        # Ridge Classifier does not give probabilities
        if not self.support_probabilities:
            self.clf = RidgeClassifierCV(alphas=np.logspace(-1, 6, 10), normalize=False)
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
        if self.use_first_differences:
            X = self._add_first_order_differences(X)

        parallel_res = Parallel(n_jobs=self._threads_to_use, backend="threading")(
            delayed(_parallel_transform_words)(X, self.SFA_transformers[ind])
            for ind in range(self.ensemble_size)
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
            "use_first_order_differences": False,
            "support_probabilities": True,
            "bigrams": False,
        }


def _compute_window_inc(series_length, window_inc):
    win_inc = window_inc
    if series_length < 100:
        win_inc = 1  # less than 100 is ok time-wise

    return win_inc


def _parallel_transform_words(X, SFA_transformers):
    # On each dimension, perform SFA

    all_words = []
    for dim in range(X.shape[1]):
        words = SFA_transformers[dim].transform(X[:, dim])
        all_words.append(words)

    return all_words


def _parallel_fit(
    ind,
    X,
    y,
    window_sizes,
    alphabet_sizes,
    word_lengths,
    norm_options,
    use_first_differences,
    binning_strategies,
    variance,
    anova,
    bigrams,
    n_jobs,
    max_feature_count,
    ensemble_size,
    feature_selection,
    random_state,
):
    if random_state is not None:
        rng = check_random_state(random_state + ind)
    else:
        rng = check_random_state(random_state)

    window_size = rng.choice(window_sizes)
    alphabet_size = rng.choice(alphabet_sizes)
    word_length = min(window_size - 2, rng.choice(word_lengths))
    norm = rng.choice(norm_options)
    first_difference = rng.choice([True, False])
    binning_strategy = rng.choice(binning_strategies)

    series_length = X.shape[-1]
    dilation = max(
        1,
        np.int32(2 ** rng.uniform(0, np.log2((series_length - 1) / (window_size - 1)))),
    )

    all_words = []
    SFA_transformers = []

    # On each dimension, perform SFA
    for dim in range(X.shape[1]):
        X_dim = X[:, dim]

        transformer = SFAFast(
            variance=variance,
            word_length=word_length,
            alphabet_size=alphabet_size,
            window_size=window_size,
            norm=norm,
            anova=anova,
            binning_method=binning_strategy,
            # remove_repeat_words=remove_repeat_words,
            bigrams=bigrams,
            dilation=dilation,
            # lower_bounding=lower_bounding,
            first_difference=first_difference,
            feature_selection=feature_selection,
            max_feature_count=max_feature_count // ensemble_size,
            random_state=ind,
            return_sparse=not (
                feature_selection == "none" and alphabet_size == 2 and not bigrams
            ),
            n_jobs=n_jobs,
        )
        all_words.append(transformer.fit_transform(X_dim, y))
        SFA_transformers.append(transformer)

    return all_words, SFA_transformers
