"""Benchmarks tests."""
from typing import Callable

import pandas as pd
import pytest

from sktime.base import BaseEstimator
from sktime.benchmarking import benchmarks
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies


def factory_estimator_class_task(**kwargs) -> Callable:
    """Return task for getting class of estimator."""

    def estimator_class_task(estimator: BaseEstimator) -> str:
        """Return class of estimator.

        Used as simple task for testing purposes.
        """
        return {"estimator_class": str(estimator.__class__)}

    return estimator_class_task


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_basebenchmark(tmp_path):
    """Test registering estimator, registering a simple task, and running."""
    benchmark = benchmarks.BaseBenchmark()

    benchmark.add_estimator(NaiveForecaster(strategy="drift"))
    benchmark._add_task(factory_estimator_class_task)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    results_df = results_df.drop(columns=["runtime_secs"])

    expected_results_df = pd.DataFrame(
        [
            (
                "factory_estimator_class_task",
                "NaiveForecaster",
                "<class 'sktime.forecasting.naive.NaiveForecaster'>",
            )
        ],
        columns=["validation_id", "model_id", "estimator_class"],
    )

    pd.testing.assert_frame_equal(expected_results_df, results_df)
    pd.testing.assert_frame_equal(
        expected_results_df, pd.read_csv(results_file).drop(columns=["runtime_secs"])
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_add_estimator_args(tmp_path):
    """Test adding estimator with args specified."""
    benchmark = benchmarks.BaseBenchmark()

    benchmark.add_estimator(
        estimator=NaiveForecaster(strategy="drift"),
        estimator_id="test_id-v1",
    )
    benchmark._add_task(factory_estimator_class_task)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    assert results_df.iloc[0, 1] == "test_id-v1"


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_add_task_args(tmp_path):
    """Test adding task with args specified."""
    benchmark = benchmarks.BaseBenchmark()

    benchmark.add_estimator(NaiveForecaster(strategy="drift"))
    benchmark._add_task(
        task_entrypoint=factory_estimator_class_task,
        task_kwargs={"some_kwarg": "some_value"},
        task_id="test_id-v1",
    )

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    assert results_df.iloc[0, 0] == "test_id-v1"


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_add_task_string_entrypoint(tmp_path):
    """Test adding task using string of entrypoint."""
    benchmark = benchmarks.BaseBenchmark()

    benchmark.add_estimator(NaiveForecaster(strategy="drift"))
    benchmark._add_task(
        "sktime.benchmarking.tests.test_benchmarks:factory_estimator_class_task"
    )

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    assert results_df.iloc[0, 3] == "<class 'sktime.forecasting.naive.NaiveForecaster'>"


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
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
