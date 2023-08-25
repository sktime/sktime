"""Canonical time warping distance, from tslearn."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.dists_kernels.base.adapters import _TslearnPwTrafoAdapter


class CtwDistTslearn(_TslearnPwTrafoAdapter, BasePairwiseTransformerPanel):
    """Canonical time warping distance, from tslearn.

    Direct interface to ``tslearn.metrics.cdist_ctw``.

    Parameters
    ----------
    max_iter : int (default: 100)
        Number of iterations for the CTW algorithm.
    n_components : int (default: None)
        Number of components to be used for Canonical Correlation Analysis.
        If None, the minimum number of features of inputs is used.
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
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    References
    ----------
    .. [1] F. Zhou and F. Torre, "Canonical time warping for alignment of
       human behavior". NIPS 2009.
    """

    _tags = {"symmetric": True, "pwtrafo_type": "distance"}

    def __init__(
        self,
        max_iter=100,
        n_components=None,
        global_constraint=None,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
        n_jobs=None,
        verbose=0,
    ):
        self.max_iter = max_iter
        self.n_components = n_components
        self.global_constraint = global_constraint
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.itakura_max_slope = itakura_max_slope
        self.n_jobs = n_jobs
        self.verbose = verbose

        super().__init__()

    def _get_tslearn_pwtrafo(self):
        """Adapter method to get tslearn pwtrafo."""
        from tslearn.metrics.ctw import cdist_ctw

        return cdist_ctw

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
        params0 = {"n_components": 2}
        params1 = {
            "max_iter": 20,
            "global_constraint": "itakura",
            "itakura_max_slope": 1.5,
            "n_components": 2,
        }
        params2 = {
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius": 2,
            "n_components": 2,
        }
        # fails with _components = None - is that a bug?

        return [params0, params1, params2]
