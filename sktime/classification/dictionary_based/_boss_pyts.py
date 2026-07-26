"""BOSS classifier, from pyts."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["BOSSVSClassifierPyts"]

from sktime.base.adapters._pyts import _PytsAdapter
from sktime.classification.base import BaseClassifier


class BOSSVSClassifierPyts(_PytsAdapter, BaseClassifier):
    """Bag-of-SFA Symbols in Vector Space, from pyts.

    Direct interface to ``pyts.classification.BOSSVS``,
    author of the interfaced class is ``johannfaouzi``.

    Each time series is transformed into an histogram using the
    Bag-of-SFA Symbols (BOSS) algorithm. Then, for each class, the histograms
    are added up and a tf-idf vector is computed. The predicted class for
    a new sample is the class giving the highest cosine similarity between
    its tf vector and the tf-idf vectors of each class.

    Parameters
    ----------
    word_size : int (default = 4)
        Size of each word.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and 26.

    window_size : int or float (default = 10)
        Size of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_size * n_timestamps)``.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_step * n_timestamps)``.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    drop_sum : bool (default = False)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    norm_mean : bool (default = False)
        If True, center each subseries before scaling.

    norm_std : bool (default = False)
        If True, scale each subseries to unit variance.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurrence of back to back
        identical occurrences of the same words.

    use_idf : bool (default = True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool (default = False)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool (default = True)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    idf_ : array, shape = (n_features,) , or None
        The learned idf vector (global term weights) when ``use_idf=True``,
        None otherwise.

    tfidf_ : array, shape = (n_classes, n_words)
        Term-document matrix.

    vocabulary_ : dict
        A mapping of feature indices to terms.

    References
    ----------
    .. [1] P. Schäfer, "Scalable Time Series Classification". Data Mining and
           Knowledge Discovery, 30(5), 1273-1298 (2016).
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["johannfaouzi", "fkiraly"],  # johannfaouzi is author of upstream
        "python_dependencies": "pyts",
        # estimator type
        # --------------
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "classifier_type": "dictionary",
    }

    # defines the name of the attribute containing the pyts estimator
    _estimator_attr = "_pyts_boss"

    def _get_pyts_class(self):
        """Get pyts class.

        should import and return pyts class
        """
        from pyts.classification import BOSSVS

        return BOSSVS

    def __init__(
        self,
        word_size=4,
        n_bins=4,
        window_size=10,
        window_step=1,
        anova=False,
        drop_sum=False,
        norm_mean=False,
        norm_std=False,
        strategy="quantile",
        alphabet=None,
        numerosity_reduction=True,
        use_idf=True,
        smooth_idf=False,
        sublinear_tf=True,
    ):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_size = window_size
        self.window_step = window_step
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.strategy = strategy
        self.alphabet = alphabet
        self.numerosity_reduction = numerosity_reduction
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        super().__init__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "word_size": 3,
            "n_bins": 3,
            "window_size": 5,
            "window_step": 2,
            "anova": True,
            "drop_sum": True,
            "norm_mean": True,
            "norm_std": True,
            "strategy": "uniform",
            "smooth_idf": True,
            "sublinear_tf": False,
        }
        return [params1, params2]
