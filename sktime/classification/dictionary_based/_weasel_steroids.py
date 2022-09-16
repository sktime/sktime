# -*- coding: utf-8 -*-
"""WEASEL on Steroids classifier.

Dictionary based classifier based on SFA transform, BOSS and linear regression.

The fastest WEASEL on earth
Tour de WEASEL
WEASEL on Steroids
WEASEL + Rocket-Science

Rocket-Science on WEASEL
"""

__author__ = ["patrickzib"]
__all__ = ["WEASEL_STEROIDS"]

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.dictionary_based import SFAFast

# from sktime.transformations.panel.rocket import MiniRocket


class WEASEL_STEROIDS(BaseClassifier):
    """Word Extraction for Time Series Classification (WEASEL).

    Overview: Input n series length m
    WEASEL is a dictionary classifier that builds a bag-of-patterns using SFA
    for different window lengths and learns a logistic regression classifier
    on this bag.

    There are these primary parameters:
            alphabet_size: alphabet size
            chi2-threshold: used for feature selection to select best words
            anova: select best l/2 fourier coefficients other than first ones
            bigrams: using bigrams of SFA words
            binning_strategy: the binning strategy used to discretise into
                             SFA words.
    WEASEL slides a window length w along the series. The w length window
    is shortened to an l length word through taking a Fourier transform and
    keeping the best l/2 complex coefficients using an anova one-sided
    test. These l coefficients are then discretised into alpha possible
    symbols, to form a word of length l. A histogram of words for each
    series is formed and stored.
    For each window-length a bag is created and all words are joint into
    one bag-of-patterns. Words from different window-lengths are
    discriminated by different prefixes.
    fit involves training a logistic regression classifier on the single
    bag-of-patterns.

    predict uses the logistic regression classifier

    Parameters
    ----------
    anova: boolean, default=True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given
    bigrams: boolean, default=True
        whether to create bigrams of SFA words
    binning_strategies: ["equi-depth", "equi-width", "information-gain"],
    default="information-gain"
        The binning method used to derive the breakpoints.
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
    MUSE

    References
    ----------
    .. [1] Patrick Schäfer and Ulf Leser, "Fast and Accurate Time Series Classification
    with WEASEL", in proc ACM on Conference on Information and Knowledge Management,
    2017, https://dl.acm.org/doi/10.1145/3132847.3132980

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/dictionary_based/WEASEL.java>`_.

    Examples
    --------
    >>> from sktime.classification.dictionary_based import WEASEL_STEROIDS
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = WEASEL_STEROIDS(window_inc=4)
    >>> clf.fit(X_train, y_train)
    WEASEL_STEROIDS(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capabilitys:multithreading": True,
        "classifier_type": "dictionary",
    }

    def __init__(
        self,
        anova=False,
        variance=True,
        bigrams=False,
        lower_bounding=True,
        binning_strategies=["equi-depth"],
        ensemble_size=50,
        max_feature_count=20_000,
        min_window=4,
        max_window=24,
        norm_options=[False],
        word_lengths=[8],
        alphabet_sizes=[4],
        use_first_differences=[True, False],
        feature_selection="random",
        remove_repeat_words=False,
        random_state=None,
        sections=1,
        n_jobs=4,
    ):
        self.alphabet_sizes = alphabet_sizes

        self.anova = anova
        self.variance = variance

        self.norm_options = norm_options
        self.word_lengths = word_lengths

        self.bigrams = bigrams
        self.lower_bounding = lower_bounding
        self.binning_strategies = binning_strategies
        self.random_state = random_state

        self.min_window = min_window
        self.max_window = max_window
        self.ensemble_size = ensemble_size
        self.max_feature_count = max_feature_count
        self.use_first_differences = use_first_differences
        self.feature_selection = feature_selection
        self.remove_repeat_words = remove_repeat_words

        self.window_sizes = []
        self.series_length = 0
        self.n_instances = 0
        self.sections = sections

        self.SFA_transformers = []

        self.clf = None
        self.n_jobs = n_jobs

        # set_num_threads(n_jobs)

        super(WEASEL_STEROIDS, self).__init__()

    def _fit(self, X, y):
        """Build a WEASEL classifiers from the training set (X, y).

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        # Window length parameter space dependent on series length
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]
        XX = X.squeeze(1)

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

        # Randomly choose window sizes
        self.window_sizes = np.arange(self.min_window, self.max_window + 1, 1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, backend="threading")(
            delayed(_parallel_fit)(
                i,
                XX,
                y.copy(),
                self.window_sizes,
                self.alphabet_sizes,
                self.word_lengths,
                self.series_length,
                self.norm_options,
                self.use_first_differences,
                self.binning_strategies,
                self.variance,
                self.anova,
                self.bigrams,
                self.lower_bounding,
                self.n_jobs,
                self.max_feature_count,
                self.ensemble_size,
                self.feature_selection,
                self.remove_repeat_words,
                self.sections,
                self.random_state,
            )
            # for i in range(self.ensemble_size)
            for i in range(len(self.window_sizes))
        )

        sfa_words = []
        # self.norms = []
        for (
            words,
            transformer,
            # norms
        ) in parallel_res:
            self.SFA_transformers.extend(transformer)
            sfa_words.extend(words)

        # self.rocket = MiniRocket(random_state=1379, n_jobs=self.n_jobs)
        # X_features = self.rocket.fit_transform(X, y)

        # merging arrays from different threads
        if type(sfa_words[0]) is np.ndarray:
            # sfa_words.append(X_features)
            all_words = np.concatenate(sfa_words, axis=1)
        else:
            # sfa_words.append(csr_matrix(X_features.values))
            all_words = hstack(sfa_words)

        self.clf = make_pipeline(
            SelectPercentile(chi2, percentile=50),
            RidgeClassifierCV(alphas=np.logspace(-1, 5, 10)),  # , normalize=True
        )  # TODO testen??

        self.clf.fit(all_words, y)
        self.total_features_count = all_words.shape[1]
        if hasattr(self.clf, "best_score_"):
            self.cross_val_score = self.clf.best_score_

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

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
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        XX = X.squeeze(1)

        parallel_res = Parallel(n_jobs=self.n_jobs, timeout=99999, backend="threading")(
            delayed(transformer.transform)(XX) for transformer in self.SFA_transformers
        )

        all_words = []
        for words in parallel_res:
            # words = words.astype(np.float32) / norm
            all_words.append(words)

        # X_features = self.rocket.transform(X)

        if type(all_words[0]) is np.ndarray:
            # all_words.append(X_features)
            all_words = np.concatenate(all_words, axis=1)
        else:
            # all_words.append(csr_matrix(X_features.values))
            all_words = hstack(all_words)

        return all_words

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
        return {}


def _parallel_fit(
    i,
    X,
    y,
    window_sizes,
    alphabet_sizes,
    word_lengths,
    series_length,
    norm_options,
    use_first_differences,
    binning_strategies,
    variance,
    anova,
    bigrams,
    lower_bounding,
    n_jobs,
    max_feature_count,
    ensemble_size,
    feature_selection,
    remove_repeat_words,
    sections,
    random_state,
):
    if random_state is None:
        rng = check_random_state(None)
    else:
        rng = check_random_state(random_state + i)

    window_size = rng.choice(window_sizes)
    # window_size = window_sizes[i]

    ii = ensemble_size / len(window_sizes)
    dilations = np.maximum(
        1,
        np.int32(
            2
            ** rng.uniform(
                0,
                np.log2((series_length - 1) / (window_size - 1)),
                size=int(np.ceil(ii)),
            )
        ),
    )
    all_words = []
    all_transformers = []

    for dilation in dilations:
        alphabet_size = rng.choice(alphabet_sizes)

        # maximize word-length
        word_length = min(window_size - 2, rng.choice(word_lengths))
        norm = rng.choice(norm_options)
        first_difference = rng.choice(use_first_differences)  # TODO always True???
        binning_strategy = rng.choice(binning_strategies)

        transformer = getSFAFast(
            alphabet_size,
            alphabet_sizes,
            anova,
            bigrams,
            binning_strategy,
            dilation,
            ensemble_size,
            feature_selection,
            first_difference,
            i,
            lower_bounding,
            max_feature_count,
            n_jobs,
            norm,
            remove_repeat_words,
            sections,
            variance,
            window_size,
            word_length,
        )

        # generate SFA words on sample
        words = transformer.fit_transform(X, y)
        all_words.append(words)
        all_transformers.append(transformer)

    return all_words, all_transformers


def getSFAFast(
    alphabet_size,
    alphabet_sizes,
    anova,
    bigrams,
    binning_strategy,
    dilation,
    ensemble_size,
    feature_selection,
    first_difference,
    i,
    lower_bounding,
    max_feature_count,
    n_jobs,
    norm,
    remove_repeat_words,
    sections,
    variance,
    window_size,
    word_length,
):
    transformer = SFAFast(
        variance=variance,
        word_length=word_length,
        alphabet_size=alphabet_size,
        window_size=window_size,
        norm=norm,
        anova=anova,
        binning_method=binning_strategy,
        remove_repeat_words=remove_repeat_words,
        bigrams=bigrams,
        dilation=dilation,
        lower_bounding=lower_bounding,
        first_difference=first_difference,
        feature_selection=feature_selection,
        sections=sections,
        max_feature_count=max_feature_count // ensemble_size,
        random_state=i,
        return_sparse=not (
            feature_selection == "none"
            and alphabet_size == 2
            and not bigrams
            and len(alphabet_sizes) == 1
        ),
        n_jobs=n_jobs,
    )
    return transformer
