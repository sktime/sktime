"""Interface for MrSEQL classifier."""

__authors__ = ["lnthach", "heerme"]  # fkiraly for the wrapper

from sktime.classification._delegate import _DelegatedClassifier


class MrSEQL(_DelegatedClassifier):
    """MrSEQL = Multiple Representations Sequence Learning classification model.

    Direct Interface to MrSEQLClassifier from mrseql.
    Note: mrseql itself is copyleft (GPL3). This interface is permissive license (BSD3).

    MrSEQL is an efficient time series classifier utilizing symbolic representations of
    time series, using SAX and SFA features.

    Parameters
    ----------
    seql_mode : str, either 'clf' or 'fs' (default).
        In the 'clf' mode, Mr-SEQL is an ensemble of SEQL models, while in the 'fs'
        mode Mr-SEQL, trains a logistic regression model with features extracted
        by SEQL from symbolic representations of time series.
    symrep : str, or list or tuple of string, strings being 'sax' or 'sfa'.
        default = "sax", i.e., only SAX features, no SFA features.
        The symbolic representations to be used to transform the input time series.
    custom_config : dict, optional, default=None
        Customized parameters for the symbolic transformation.
        If defined, symrep will be ignored.
        (no documentation of this parameter is provided in the original mrseql code)

    References
    ----------
    .. [1] Thach Le Nguyen, Severin Gsponer, Iulia Ilie, Martin O'Reilly,
       Georgiana Ifrim.
       "Interpretable Time Series Classification Using Linear Models and
       Multi-resolution Multi-domain Symbolic Representations",
       Data Mining and Knowledge Discovery, 2019.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["lnthach", "heerme", "fkiraly"],
        "maintainers": ["lnthach", "heerme", "fkiraly"],
        "python_dependencies": "mrseql",
        "requires_cython": True,
        # estimator type
        # --------------
        "X_inner_mtype": "nested_univ",
    }

    def __init__(
        self,
        seql_mode="fs",
        symrep="sax",
        custom_config=None,
    ):
        self.seql_mode = seql_mode
        self.symrep = symrep
        self.custom_config = custom_config

        if not isinstance(symrep, (list, tuple)):
            self._symrep = [symrep]
        else:
            self._symrep = symrep

        if "sfa" in self._symrep:
            self.set_tags(**{"python_dependencies": ["mrseql", "numba"]})

        super().__init__()

        # construct the delegate - direct delegation to MrSEQLClassifier
        from mrseql import MrSEQLClassifier

        kwargs = self.get_params(deep=False)
        kwargs["symrep"] = self._symrep

        self.estimator_ = MrSEQLClassifier(**kwargs)

    # temporary workaround - delegate is not sktime interface compliant,
    # does not implement get_fitted_params
    # see https://github.com/mlgig/mrseql/issues/7
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
        from sktime.utils.dependencies import _check_soft_dependencies

        params1 = {}

        if not _check_soft_dependencies("numba", severity="none"):
            symrep = ["sax"]
        else:
            symrep = ["sax", "sfa"]

        params2 = {
            "seql_mode": "fs",
            "symrep": symrep,
        }

        return [params1, params2]
