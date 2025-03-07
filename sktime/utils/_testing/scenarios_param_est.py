"""Test scenarios for parameter estimators.

Contains TestScenario concrete children to run in tests for parameter estimators.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_param_est"]

from inspect import isclass

from sktime.base import BaseObject
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42
RAND_SD2 = 84


def get_tag(obj, tag_name):
    """Shorthand for get_tag vs get_class_tag, obj can be class or object."""
    if isclass(obj):
        return obj.get_class_tag(tag_name)
    else:
        return obj.get_tag(tag_name)


# no logic in scenario classes, but placeholder and for pattern homogeneity
class ParamFitterTestScenario(TestScenario, BaseObject):
    """Generic test scenario for aligners."""

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
        # pre-refactor classes can't deal with Series *and* Panel both
        obj_has_c_multivariate = get_tag(obj, "capability:multivariate")
        scenario_is_multivariate = not self.get_tag("X_univariate")
        if not obj_has_c_multivariate and scenario_is_multivariate:
            return False

        return True


class ParamFitterUnivariate(ParamFitterTestScenario):
    """Estimate parameters on a univariate series."""

    _tags = {"X_univariate": True, "pairwise": False, "is_enabled": True}

    @property
    def args(self):
        return {
            "fit": {
                "X": _make_series(n_timepoints=30, n_columns=1, random_state=RAND_SEED)
            },
        }

    default_method_sequence = ["fit", "get_fitted_params"]


class ParamFitterMultivariate(ParamFitterTestScenario):
    """Estimate parameters on a multivariate series."""

    _tags = {"X_univariate": False, "pairwise": False, "is_enabled": True}

    @property
    def args(self):
        return {
            "fit": {
                "X": _make_series(n_timepoints=30, n_columns=2, random_state=RAND_SEED)
            },
        }

    default_method_sequence = ["fit", "get_fitted_params"]


class ParamFitterPairwiseUnivariate(ParamFitterTestScenario):
    """Estimate parameters on two univariate series."""

    _tags = {"X_univariate": True, "pairwise": True, "is_enabled": True}

    @property
    def args(self):
        return {
            "fit": {
                "X": _make_series(n_timepoints=30, n_columns=1, random_state=RAND_SEED),
                "y": _make_series(n_timepoints=30, n_columns=1, random_state=RAND_SD2),
            },
        }

    default_method_sequence = ["fit", "get_fitted_params"]


scenarios_param_est = [
    ParamFitterUnivariate,
    ParamFitterMultivariate,
    ParamFitterPairwiseUnivariate,
]
