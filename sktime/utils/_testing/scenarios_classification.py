# -*- coding: utf-8 -*-
"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressirs.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_classification", "scenarios_regression"]

from copy import deepcopy

from sktime.base import BaseObject
from sktime.utils._testing.panel import _make_classification_y, _make_panel_X
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


class ClassifierTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

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
        if key in ["predict_proba", "decision_function"]:
            key = "predict"

        args = self.args[key]

        if deepcopy_args:
            args = deepcopy(args)

        return args


y = _make_classification_y(n_instances=10, random_state=RAND_SEED)
X = _make_panel_X(n_instances=10, n_timepoints=20, random_state=RAND_SEED, y=y)
X_test = _make_panel_X(n_instances=5, n_timepoints=20, random_state=RAND_SEED)


class ClassifierFitPredict(ClassifierTestScenario):
    """Fit/predict with panel X and labels y."""

    _tags = {"X_univariate": True, "pre-refactor": True}

    args = {
        "fit": {"y": y, "X": X},
        "predict": {"X": X_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


scenarios_classification = [
    ClassifierFitPredict,
]

# we use the same scenarios for regression, as in the old test suite
scenarios_regression = [
    ClassifierFitPredict,
]
