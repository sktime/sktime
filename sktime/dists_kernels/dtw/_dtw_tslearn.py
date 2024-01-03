"""Dynamic time warping distance, from tslearn."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.dists_kernels.base.adapters import _TslearnPwTrafoAdapter


class DtwDistTslearn(_TslearnPwTrafoAdapter, BasePairwiseTransformerPanel):
    """Dynamic time warping distance, from tslearn.

    Direct interface to ``tslearn.metrics.cdist_dtw``.

    Parameters
    ----------
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and ``global_constraint`` is set to ``"sakoe_chiba"``, a radius of
        1 is used.
        If both ``sakoe_chiba_radius`` and ``itakura_max_slope`` are set,
        ``global_constraint`` is used to infer which constraint to use among the
        two. In this case, if ``global_constraint`` corresponds to no global
        constraint, a ``RuntimeWarning`` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and ``global_constraint`` is set to ``"itakura"``, a maximum slope
        of 2 is used.
        If both ``sakoe_chiba_radius`` and ``itakura_max_slope`` are set,
        ``global_constraint`` is used to infer which constraint to use among the
        two. In this case, if ``global_constraint`` corresponds to no global
        constraint, a ``RuntimeWarning`` is raised and no global constraint is
        used.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """

    _tags = {"symmetric": True, "pwtrafo_type": "distance"}

    def __init__(
        self,
        global_constraint=None,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
        n_jobs=None,
        verbose=0,
    ):
        self.global_constraint = global_constraint
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.itakura_max_slope = itakura_max_slope
        self.n_jobs = n_jobs
        self.verbose = verbose

        super().__init__()

    def _get_tslearn_pwtrafo(self):
        """Adapter method to get tslearn pwtrafo."""
        from tslearn.metrics.dtw_variants import cdist_dtw

        return cdist_dtw

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
        params0 = {}
        params1 = {"global_constraint": "itakura", "itakura_max_slope": 1.5}
        params2 = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 2}

        return [params0, params1, params2]


class SoftDtwDistTslearn(_TslearnPwTrafoAdapter, BasePairwiseTransformerPanel):
    """Soft dynamic time warping distance, from tslearn.

    Direct interface to ``tslearn.metrics.cdist_soft_dtw`` and
    ``tslearn.metrics.cdist_soft_dtw_normalized``.

    Parameters
    ----------
    normalized : bool, default = False
        Whether the DTW distance should be normalized.
        If ``False``, interfaces ``tslearn.metrics.cdist_soft_dtw``.
        If ``True``, interfaces ``tslearn.metrics.cdist_soft_dtw_normalized``.
    gamma : float (default 1.)
        Gamma parameter for Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """

    _tags = {
        "symmetric": True,
        "pwtrafo_type": "distance",
        "python_dependencies": "tslearn>=0.6.2",
    }

    _inner_params = ["gamma"]

    def __init__(self, normalized=False, gamma=1.0):
        self.normalized = normalized
        self.gamma = gamma

        super().__init__()

    def _get_tslearn_pwtrafo(self):
        """Adapter method to get tslearn pwtrafo."""
        from tslearn.metrics.softdtw_variants import (
            cdist_soft_dtw,
            cdist_soft_dtw_normalized,
        )

        if self.normalized:
            return cdist_soft_dtw_normalized
        else:
            return cdist_soft_dtw

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
        params0 = {}
        params1 = {"normalized": True, "gamma": 1.5}
        params2 = {"gamma": 0.5}

        return [params0, params1, params2]
