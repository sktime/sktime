"""Longest Common Subsequence similarity distance, from tslearn."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.dists_kernels.base.adapters import _TslearnPwTrafoAdapter


class LcssTslearn(_TslearnPwTrafoAdapter, BasePairwiseTransformerPanel):
    """Longest Common Subsequence similarity distance, from tslearn.

    Direct interface to ``tslearn.metrics.lcss``.

    Parameters
    ----------
    eps : float (default: 1.)
        Maximum matching distance threshold.
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

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["danisodu", "rtavenar", "fkiraly"],
        "python_dependencies": ["tslearn"],
        # estimator type
        # --------------
        "symmetric": True,
        "pwtrafo_type": "distance",
    }

    _is_cdist = False

    def __init__(
        self,
        eps=1.0,
        global_constraint=None,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
    ):
        self.eps = eps
        self.global_constraint = global_constraint
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.itakura_max_slope = itakura_max_slope

        super().__init__()

    def _get_tslearn_pwtrafo(self):
        """Adapter method to get tslearn pwtrafo."""
        from tslearn.metrics import lcss

        return lcss

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
        params0 = {}
        params1 = {"eps": 0.9, "global_constraint": "itakura", "itakura_max_slope": 1.5}
        params2 = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 2}

        return [params0, params1, params2]
