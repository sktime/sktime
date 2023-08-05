"""Global Alignment Kernel from tslearn."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.dists_kernels.base.adapters import _TslearnPwTrafoAdapter


class GAKernel(_TslearnPwTrafoAdapter, BasePairwiseTransformerPanel):
    r"""Global Alignment Kernel, from tslearn.

    Direct interface to ``tslearn.metrics.cdist_gak`` and ``gak``.

    Implements the fast GAK from [1]_.

    Parameters
    ----------
    sigma : float, default 1.
        Bandwidth of the internal gaussian kernel used for GAK
    normalized : bool, optional, default=True
        if True, computes the normalized GAK (``cdist_gak``)
        if False, computes unnormalized GAK (``gak``)
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.
    verbose : int, optional, default=0
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """

    _tags = {"symmetric": True, "pwtrafo_type": "kernel"}

    def __init__(
        self,
        sigma=1.0,
        normalized=True,
        n_jobs=None,
        verbose=0,
    ):
        self.sigma = sigma
        self.normalized = normalized
        self.n_jobs = n_jobs
        self.verbose = verbose

        super().__init__()

    _inner_params = ["sigma", "n_jobs", "verbose"]

    def _get_tslearn_pwtrafo(self):
        """Adapter method to get tslearn pwtrafo."""
        if self.normalized:
            from tslearn.metrics.softdtw_variants import cdist_gak

            return cdist_gak
        else:
            from tslearn.metrics.softdtw_variants import gak

            return gak

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for distance/kernel transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params0 = {"normalized": True, "sigma": 0.5}
        params1 = {"normralized": False}

        return [params0, params1]
