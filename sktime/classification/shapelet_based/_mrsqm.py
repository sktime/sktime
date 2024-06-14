"""Interface for MrSQM classifier."""

__authors__ = ["lnthach", "heerme"]  # fkiraly for the wrapper

from sktime.classification._delegate import _DelegatedClassifier


class MrSQM(_DelegatedClassifier):
    """MrSQM = Multiple Representations Sequence Miner.

    Direct Interface to MrSQMClassifier from mrsqm.
    Note: mrsqm itself is copyleft (GPL3). This interface is permissive license (BSD3).

    MrSQM is an efficient time series classifier utilizing symbolic representations of
    time series. MrSQM implements four different feature selection strategies =
    (R,S,RS,SR) that can quickly select subsequences from multiple symbolic
    representations of time series data.

    Parameters
    ----------
    strat               : str, one of 'R','S','SR', or 'RS', default="RS"
        feature selection strategy. By default set to 'RS'.
        R and S are single-stage filters while RS and SR are two-stage filters.
    features_per_rep    : int, default=500
        (maximum) number of features selected per representation.
    selection_per_rep   : int, default=2000
        (maximum) number of candidate features selected per representation.
        Only applied in two stages strategies (RS and SR), otherwise ignored.
    nsax                : int, default=1
        number of representations produced by sax transformation.
    nsfa                : int, default=0
        number of representations produced by sfa transformation.
    custom_config       : dict, default=None
        customized parameters for the symbolic transformation.
    random_state        : int, default=None.
        random seed for the classifier.
    sfa_norm            : bool, default=True.
        whether to apply time series normalisation (standardisation).

    References
    ----------
    .. [1] Thach Le Nguyen and Georgiana Ifrim.
       "MrSQM: Fast Time Series Classification with Symbolic Representations
       and Efficient Sequence Mining" arXiv preprint arXiv:2109.01036 (2021).
    .. [2] Thach Le Nguyen and Georgiana Ifrim.
       "Fast Time Series Classification with Random Symbolic Subsequences".
       AALTD 2022.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["lnthach", "heerme", "fkiraly"],
        "maintainers": ["lnthach", "heerme", "fkiraly"],
        "python_dependencies": "mrsqm",
        "requires_cython": True,
        # estimator type
        # --------------
        "X_inner_mtype": "nested_univ",
    }

    def __init__(
        self,
        strat="RS",
        features_per_rep=500,
        selection_per_rep=2000,
        nsax=1,
        nsfa=0,
        custom_config=None,
        random_state=None,
        sfa_norm=True,
    ):
        self.strat = strat
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        self.nsax = nsax
        self.nsfa = nsfa
        self.custom_config = custom_config
        self.random_state = random_state
        self.sfa_norm = sfa_norm
        super().__init__()

        # construct the delegate - direct delegation to MrSQMClassifier
        from mrsqm import MrSQMClassifier

        kwargs = self.get_params(deep=False)
        self.estimator_ = MrSQMClassifier(**kwargs)

    # temporary workaround - delegate is not sktime interface compliant,
    # does not implement get_fitted_params
    # see https://github.com/mlgig/mrsqm/issues/7
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        return {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {}

        params2 = {
            "strat": "SR",
            "features_per_rep": 200,
            "selection_per_rep": 1000,
            "nsax": 2,
            "nsfa": 1,
            "sfa_norm": False,
        }

        return [params1, params2]
