"""Test scenarios for detector estimators.

Contains TestScenario concrete children to run in tests for detector estimators.
"""

__all__ = ["scenarios_detectors"]

from inspect import isclass

from sktime.base import BaseObject
from sktime.utils._testing.scenarios import TestScenario
from sktime.utils._testing.series import _make_series

# Random seeds for reproducibility
RAND_SEED = 42
RAND_SD2 = 84


def get_tag(obj, tag_name):
    """Shorthand for get_tag vs get_class_tag, obj can be class or object."""
    if isclass(obj):
        return obj.get_class_tag(tag_name)
    else:
        return obj.get_tag(tag_name)


class DetectorTestScenario(TestScenario, BaseObject):
    """Generic test scenario for detectors."""

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj."""
        obj_has_c_multivariate = get_tag(obj, "capability:multivariate")
        scenario_is_multivariate = not self.get_tag("X_univariate")
        if not obj_has_c_multivariate and scenario_is_multivariate:
            return False
        return True


class DetectorUnivariateSimple(DetectorTestScenario):
    """Simple univariate detector scenario with default settings."""

    _tags = {
        "is_enabled": True,
        "scitype": "detector",
        "X_univariate": True,
    }

    @property
    def args(self):
        Z = _make_series(n_timepoints=20, random_state=RAND_SEED)
        return {
            "fit": {"X": Z},
            "predict": {"X": Z},
            "transform": {"X": Z},
        }

    default_method_sequence = ["fit", "predict", "transform"]


# List of scenarios to be imported by tests
scenarios_detectors = [DetectorUnivariateSimple]
