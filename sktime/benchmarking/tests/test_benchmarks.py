"""Benchmarks tests."""

import pytest

from sktime.benchmarking import benchmarks
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_raise_id_restraint():
    """Test to ensure ID format is raised for malformed input ID."""
    # format of the form [username/](entity-name)-v(major).(minor)
    id_format = r"^(?:[\w:-]+\/)?([\w:.\-{}=\[\]]+)-v([\d.]+)$"
    error_msg = "Attempted to register malformed entity ID"
    benchmark = benchmarks.BaseBenchmark(id_format)
    with pytest.raises(ValueError) as exc_info:
        benchmark.add_estimator(NaiveForecaster(), "test_id")
    assert exc_info.type is ValueError, "Must raise a ValueError"
    assert error_msg in exc_info.value.args[0], "Error msg is not raised"
