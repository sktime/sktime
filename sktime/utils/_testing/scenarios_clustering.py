# -*- coding: utf-8 -*-
"""Test scenarios for clustering.

Contains TestScenario concrete children to run in tests for clusterers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_clustering"]

from copy import deepcopy

from sktime.base import BaseObject
from sktime.utils._testing.panel import _make_panel_X, make_clustering_problem
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


# no logic in scenario classes, but placeholder and for pattern homogeneity
class ClustererTestScenario(TestScenario, BaseObject):
    """Generic test scenario for clusterers."""

    def get_args(self, key, obj=None, deepcopy_args=True):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        # use same args for predict-like functions as for predict
        if key in ["predict_proba"]:
            key = "predict"

        args = self.args[key]

        if deepcopy_args:
            args = deepcopy(args)

        return args


class ClustererFitPredict(ClustererTestScenario):
    """Fit/predict with panel Xmake_clustering_problem."""

    _tags = {"X_univariate": True, "is_enabled": True}

    args = {
        "fit": {"X": make_clustering_problem(random_state=RAND_SEED)},
        "predict": {"X": _make_panel_X(random_state=RAND_SEED)},
    }
    default_method_sequence = ["fit", "predict"]


scenarios_clustering = [
    ClustererFitPredict,
]
