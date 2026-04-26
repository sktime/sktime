"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressirs.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_bootstrap"]

from inspect import isclass

import numpy as np
from skbase.base import BaseObject

from tsbootstrap.tests.scenarios.scenarios import TestScenario

RAND_SEED = 42

rng = np.random.default_rng(RAND_SEED)


class _BootstrapTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """

        def get_tag(obj, tag_name):
            if isclass(obj):
                return obj.get_class_tag(tag_name)
            else:
                return obj.get_tag(tag_name)

        def scitype(obj):
            type_tag = obj.get_class_tag("object_type", "object")
            return type_tag

        if scitype(obj) != "bootstrap":
            return False

        is_multivariate = not self.get_tag(
            "X_univariate", False, raise_error=False
        )

        obj_can_handle_multivariate = get_tag(obj, "capability:multivariate")

        if is_multivariate and not obj_can_handle_multivariate:
            return False

        return True


X_np_uni = rng.random((20, 1))
X_np_mult = rng.random((20, 2))
exog_np = rng.random((20, 3))


class BootstrapBasicUnivar(_BootstrapTestScenario):
    """Simple call, only endogenous data."""

    _tags = {
        "X_univariate": True,
        "exog_present": False,
        "return_index": False,
    }

    args = {"bootstrap": {"X": X_np_uni}}
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapExogUnivar(_BootstrapTestScenario):
    """Call with endogenous and exogenous data."""

    _tags = {
        "X_univariate": True,
        "exog_present": True,
        "return_index": False,
    }

    args = {"bootstrap": {"X": X_np_uni, "y": exog_np}}
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapUnivarRetIx(_BootstrapTestScenario):
    """Call with endogenous and exogenous data, and query to return index."""

    _tags = {
        "X_univariate": True,
        "exog_present": True,
        "return_index": True,
    }

    args = {
        "bootstrap": {"X": X_np_uni, "y": exog_np, "return_indices": True},
        "get_n_bootstraps": {"X": X_np_uni, "y": exog_np},
    }
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapBasicMultivar(_BootstrapTestScenario):
    """Simple call, only endogenous data."""

    _tags = {
        "X_univariate": False,
        "exog_present": False,
        "return_index": False,
    }

    args = {"bootstrap": {"X": X_np_mult}}
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapExogMultivar(_BootstrapTestScenario):
    """Call with endogenous and exogenous data."""

    _tags = {
        "X_univariate": False,
        "exog_present": True,
        "return_index": False,
    }

    args = {"bootstrap": {"X": X_np_mult, "y": exog_np}}
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


class BootstrapMultivarRetIx(_BootstrapTestScenario):
    """Call with endogenous and exogenous data, and query to return index."""

    _tags = {
        "X_univariate": False,
        "exog_present": True,
        "return_index": True,
    }

    args = {
        "bootstrap": {"X": X_np_mult, "y": exog_np, "return_indices": True},
        "get_n_bootstraps": {"X": X_np_mult, "y": exog_np},
    }
    default_method_sequence = ["bootstrap", "get_n_bootstraps"]
    default_arg_sequence = ["bootstrap", "bootstrap"]


scenarios_bootstrap = [
    BootstrapBasicUnivar,
    BootstrapExogUnivar,
    BootstrapUnivarRetIx,
    BootstrapBasicMultivar,
    BootstrapExogMultivar,
    BootstrapMultivarRetIx,
]
