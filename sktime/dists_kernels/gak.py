"""Global Alignment Kernel from tslearn."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.dists_kernels.base.adapters import _TslearnPwTrafoAdapter


class GAKernel(_TslearnPwTrafoAdapter, BasePairwiseTransformerPanel):
    r"""Global Alignment Kernel, from tslearn.

    Direct interface to ``tslearn.metrics.cdist_gak``.

    Implements the fast GAK from [1]_.

    Parameters
    ----------
    sigma : float, default 1.
        Bandwidth of the internal gaussian kernel used for GAK
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    random_state : int, RandomState instance or None, optional, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    verbose : int, optional, default=0
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rtavenar", "yanncabanes", "fkiraly"],
        # rtavenar, yanncabanes credit for interfaced code
        "python_dependencies": ["tslearn"],
        # estimator type
        # --------------
        "symmetric": True,
        "pwtrafo_type": "kernel",
        "capability:random_state": True,
    }

    _inner_params = ["sigma", "n_jobs", "verbose"]

    def __init__(
        self,
        sigma=1.0,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.sigma = sigma
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

        if self.sigma == "auto":
            self.set_tags(**{"fit_is_empty": False})

    def _get_tslearn_pwtrafo(self):
        """Adapter method to get tslearn pwtrafo."""
        from tslearn.metrics.softdtw_variants import cdist_gak

        return cdist_gak

    def _fit(self, X, X2=None):
        """Fit the distance parameters if required."""
        if self.sigma == "auto":
            from tslearn.metrics import sigma_gak

            X_tslearn = self._coerce_df_list_to_list_of_arr(X)
            self.sigma_ = float(sigma_gak(X_tslearn, random_state=self.random_state))
            if self.sigma_ == 0.0:
                self.sigma_ = 1.0
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for distance/kernel transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {"sigma": 0.5}
        params1 = {"sigma": 2}
        params2 = {"sigma": "auto"}

        return [params0, params1, params2]
