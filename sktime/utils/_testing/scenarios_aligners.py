"""Test scenarios for aligners.

Contains TestScenario concrete children to run in tests for alignment algorithms.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_aligners"]

from sktime.base import BaseObject
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


# no logic in scenario classes, but placeholder and for pattern homogeneity
class AlignerTestScenario(TestScenario, BaseObject):
    """Generic test scenario for aligners."""

    pass


class AlignerPairwiseMultivariateEqual(AlignerTestScenario):
    """Align multivariate series, pairwise alignment, equal length."""

    _tags = {
        "X_univariate": False,
        "pairwise": True,
        "equal_length": True,
        "is_enabled": True,
    }

    args = {
        "fit": {
            "X": [
                _make_series(n_timepoints=20, n_columns=2, random_state=RAND_SEED),
                _make_series(n_timepoints=20, n_columns=2, random_state=RAND_SEED),
            ],
        },
    }
    default_method_sequence = ["fit"]


class AlignerPairwiseUnivariateUnequal(AlignerTestScenario):
    """Align univariate series, pairwise alignment, unequal length."""

    _tags = {
        "X_univariate": True,
        "pairwise": True,
        "equal_length": False,
        "is_enabled": False,
    }

    args = {
        "fit": {
            "X": [
                _make_series(n_timepoints=20, n_columns=1, random_state=RAND_SEED),
                _make_series(n_timepoints=30, n_columns=1, random_state=RAND_SEED),
            ],
        },
    }
    default_method_sequence = ["fit"]


class AlignerMultipleUnivariateUnequal(AlignerTestScenario):
    """Align univariate series, multiple alignment, unequal length."""

    _tags = {
        "X_univariate": True,
        "pairwise": False,
        "equal_length": False,
        "is_enabled": False,
    }

    args = {
        "fit": {
            "X": [
                _make_series(n_timepoints=20, n_columns=1, random_state=RAND_SEED),
                _make_series(n_timepoints=30, n_columns=1, random_state=RAND_SEED),
                _make_series(n_timepoints=25, n_columns=1, random_state=RAND_SEED),
            ],
        },
    }
    default_method_sequence = ["fit"]


scenarios_aligners = [
    AlignerPairwiseMultivariateEqual,
    AlignerPairwiseUnivariateUnequal,
    AlignerMultipleUnivariateUnequal,
]
