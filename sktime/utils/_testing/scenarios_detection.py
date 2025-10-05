"""Test scenarios for detectors.

Defines minimal test scenarios so that detector estimators
can be tested in a consistent way across the library.
"""

__all__ = ["DetectorUnivariateSimple", "scenarios_detectors"]

from sktime.utils._testing.scenarios import TestScenario
from sktime.utils._testing.series import _make_series

# Random seed for reproducibility
RAND_SEED = 42


class DetectorUnivariateSimple(TestScenario):
    """Simple univariate detector scenario with default settings.

    This provides a minimal example of detector input data (Z)
    to be used in smoke or unit tests for detector estimators.
    """

    _tags = {"is_enabled": True, "scitype": "detector"}

    @property
    def args(self):
        # Z represents a univariate time series input
        Z = _make_series(n_timepoints=20, random_state=RAND_SEED)
        return {"fit": {"X": Z}}

    # Defines the default sequence of methods to call
    default_method_sequence = ["fit"]


# List of scenarios to be imported by tests
scenarios_detectors = [DetectorUnivariateSimple]
