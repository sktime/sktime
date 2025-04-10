"""Forecasting benchmarks tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from sktime.benchmarking.benchmarks import coerce_estimator_and_id
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.datasets import load_airline, load_longley
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredPercentageError,
)
from sktime.split import ExpandingWindowSplitter, InstanceSplitter, SingleWindowSplitter
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.dependencies import _check_soft_dependencies

# TODO:
# Manual test is labor intensive, need to refactor the tests for fast iteration
EXPECTED_RESULTS_1 = pd.DataFrame(
    data={
        "validation_id": "[dataset=data_loader_simple]_"
        + "[cv_splitter=ExpandingWindowSplitter]",
        "model_id": "NaiveForecaster",
        "MeanSquaredPercentageError_fold_0_test": 0.0,
        "MeanSquaredPercentageError_fold_1_test": 0.111,
        "MeanSquaredPercentageError_mean": 0.0555,
        "MeanSquaredPercentageError_std": 0.0785,
    },
    index=[0],
)
EXPECTED_RESULTS_2 = pd.DataFrame(
    data={
        "validation_id": "[dataset=data_loader_simple]_"
        + "[cv_splitter=ExpandingWindowSplitter]",
        "model_id": "NaiveForecaster",
        "MeanAbsolutePercentageError_fold_0_test": 0.0,
        "MeanAbsolutePercentageError_fold_1_test": 0.333,
        "MeanAbsolutePercentageError_mean": 0.1666,
        "MeanAbsolutePercentageError_std": 0.2357,
        "MeanAbsoluteError_fold_0_test": 0.0,
        "MeanAbsoluteError_fold_1_test": 1.0,
        "MeanAbsoluteError_mean": 0.5,
        "MeanAbsoluteError_std": 0.7071,
    },
    index=[0],
)

EXPECTED_RESULTS_GLOBAL_1 = pd.DataFrame(
    data={
        "validation_id": "[dataset=data_loader_global]_"
        + "[cv_splitter=SingleWindowSplitter]_[cv_global=InstanceSplitter]",
        "model_id": "PytorchForecastingDeepAR",
        "MeanSquaredPercentageError_fold_0_test": 0.0,
        "MeanSquaredPercentageError_fold_1_test": 0.0,
        "MeanSquaredPercentageError_mean": 0.0,
        "MeanSquaredPercentageError_std": 0.0,
    },
    index=[0],
)
EXPECTED_RESULTS_GLOBAL_2 = pd.DataFrame(
    data={
        "validation_id": "[dataset=data_loader_global]_"
        + "[cv_splitter=SingleWindowSplitter]_[cv_global=InstanceSplitter]",
        "model_id": "PytorchForecastingDeepAR",
        "MeanAbsolutePercentageError_fold_0_test": 0.0,
        "MeanAbsolutePercentageError_fold_1_test": 0.0,
        "MeanAbsolutePercentageError_mean": 0.0,
        "MeanAbsolutePercentageError_std": 0.0,
        "MeanAbsoluteError_fold_0_test": 0.0,
        "MeanAbsoluteError_fold_1_test": 0.0,
        "MeanAbsoluteError_mean": 0.0,
        "MeanAbsoluteError_std": 0.0,
    },
    index=[0],
)

COER_CASES = [
    (
        NaiveForecaster(),
        "NaiveForecaster",
        {"NaiveForecaster": NaiveForecaster()},
    ),
    (NaiveForecaster(), None, {"NaiveForecaster": NaiveForecaster()}),
    (
        [NaiveForecaster(), TrendForecaster()],
        None,
        {
            "NaiveForecaster": NaiveForecaster(),
            "TrendForecaster": TrendForecaster(),
        },
    ),
    (
        {"estimator_1": NaiveForecaster()},
        None,
        {"estimator_1": NaiveForecaster()},
    ),
]


def data_loader_simple() -> pd.DataFrame:
    """Return simple data for use in testing."""
    return pd.DataFrame([2, 2, 3])


def data_loader_global():
    """Return simple data for use in global mode testing."""
    hierarchy_levels = (4, 4)
    timepoints = 10
    data = _make_hierarchical(
        hierarchy_levels=hierarchy_levels,
        max_timepoints=timepoints,
        min_timepoints=timepoints,
        n_columns=2,
    )
    for col in data.columns:
        data[col] = np.ones(timepoints * np.prod(hierarchy_levels))
    x = data["c0"].to_frame()
    y = data["c1"].to_frame()
    return x, y


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize(
    "expected_results_df, scorers",
    [
        (EXPECTED_RESULTS_1, [MeanSquaredPercentageError()]),
        (EXPECTED_RESULTS_2, [MeanAbsolutePercentageError(), MeanAbsoluteError()]),
    ],
)
def test_forecastingbenchmark(tmp_path, expected_results_df, scorers):
    """Test benchmarking a forecaster estimator."""
    benchmark = ForecastingBenchmark()

    benchmark.add_estimator(NaiveForecaster(strategy="last"))

    cv_splitter = ExpandingWindowSplitter(
        initial_window=1,
        step_length=1,
        fh=1,
    )
    benchmark.add_task(data_loader_simple, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    results_df = results_df.drop(
        columns=[
            "fit_time_fold_0_test",
            "pred_time_fold_0_test",
            "fit_time_fold_0_test",
            "pred_time_fold_0_test",
            "fit_time_mean",
            "fit_time_std",
            "pred_time_mean",
            "pred_time_std",
        ]
    )

    results_df = results_df[expected_results_df.columns]

    pd.testing.assert_frame_equal(
        expected_results_df, results_df, check_exact=False, atol=0, rtol=0.001
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.skipif(
    not _check_soft_dependencies("pytorch-forecasting", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "expected_results_df, scorers",
    [
        (EXPECTED_RESULTS_GLOBAL_1, [MeanSquaredPercentageError()]),
        (
            EXPECTED_RESULTS_GLOBAL_2,
            [MeanAbsolutePercentageError(), MeanAbsoluteError()],
        ),
    ],
)
def test_forecastingbenchmark_global_mode(
    tmp_path,
    expected_results_df,
    scorers,
):
    """Test benchmarking a forecaster estimator in gloabl mode."""
    from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR

    benchmark = ForecastingBenchmark()

    params = {
        "trainer_params": {
            # the training process is not deterministic
            # train 10 epoches to make sure loss is low enough
            "max_epochs": 1,
        },
        "model_params": {
            "cell_type": "GRU",
            "rnn_layers": 1,
            "hidden_size": 2,
            "log_interval": -1,
        },
        "dataset_params": {
            "max_encoder_length": 2,
        },
        "random_log_path": True,  # fix parallel file access error in CI
    }
    benchmark.add_estimator(PytorchForecastingDeepAR(**params))

    benchmark.add_task(
        data_loader_global,
        SingleWindowSplitter(fh=[1], window_length=4),
        scorers,
        cv_global=InstanceSplitter(KFold(2)),
    )

    results_file = tmp_path / "results_global_mode.csv"
    results_df = benchmark.run(results_file)
    results_df = results_df.drop(
        columns=[
            "runtime_secs",
            "fit_time_fold_0_test",
            "pred_time_fold_0_test",
            "fit_time_fold_0_test",
            "pred_time_fold_0_test",
            "fit_time_mean",
            "fit_time_std",
            "pred_time_mean",
            "pred_time_std",
            "fit_time_fold_1_test",
            "pred_time_fold_1_test",
            "fit_time_fold_1_test",
            "pred_time_fold_1_test",
        ]
    )

    pd.testing.assert_frame_equal(
        expected_results_df, results_df, check_exact=False, atol=1, rtol=1
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("estimator, estimator_id, expected_output", COER_CASES)
def test_coerce_estimator_and_id(estimator, estimator_id, expected_output):
    """Test coerce_estimator_and_id return expected output."""
    assert coerce_estimator_and_id(estimator, estimator_id) == expected_output, (
        "coerce_estimator_and_id does not return the expected output."
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize(
    "estimators",
    [
        ({"N": NaiveForecaster(), "T": TrendForecaster()}),
        ([NaiveForecaster(), TrendForecaster()]),
    ],
)
def test_multiple_estimators(estimators):
    """Test add_estimator with multiple estimators."""
    # single estimator test is checked in test_forecastingbenchmark
    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(estimators)
    registered_estimators = benchmark.estimators.entities.keys()
    assert len(registered_estimators) == len(estimators), (
        "add_estimator does not register all estimators."
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_dataset_different_format(tmp_path):
    """Test to check different dataset formats to output identical results."""
    y_1, X_1 = load_longley()
    y_2 = load_airline()
    datasets = [[load_longley, load_airline], [(y_1, X_1), (y_2, None)]]
    scorers = [MeanAbsolutePercentageError(), MeanAbsoluteError()]
    cv_splitters = [
        ExpandingWindowSplitter(initial_window=14),
        ExpandingWindowSplitter(initial_window=142),
    ]

    outputs = []
    benchmarks = [ForecastingBenchmark(), ForecastingBenchmark()]
    for idx, benchmark in enumerate(benchmarks):
        benchmark.add_estimator(NaiveForecaster(strategy="last"))
        benchmark.add_task(datasets[idx][0], cv_splitters[0], scorers, f"{idx}-1")
        benchmark.add_task(datasets[idx][1], cv_splitters[1], scorers, f"{idx}-2")
        results_file = tmp_path / f"results_{idx}.csv"
        output_df = benchmark.run(results_file)
        outputs.append(
            output_df.drop(
                columns=[
                    "runtime_secs",
                    "validation_id",
                    "fit_time_mean",
                    "pred_time_mean",
                    "fit_time_std",
                    "pred_time_std",
                    "fit_time_fold_0_test",
                    "pred_time_fold_0_test",
                    "fit_time_fold_1_test",
                    "pred_time_fold_1_test",
                ]
            )
        )

    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)


def test_add_estimator_twice(tmp_path):
    """Test adding the same estimator twice."""
    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(NaiveForecaster(strategy="last"))
    benchmark.add_estimator(NaiveForecaster(strategy="last"))
    scorers = [MeanAbsolutePercentageError()]

    cv_splitter = ExpandingWindowSplitter(
        initial_window=1,
        step_length=1,
        fh=1,
    )
    benchmark.add_task(data_loader_simple, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["NaiveForecaster", "NaiveForecaster_2"], name="model_id"),
        results_df["model_id"],
    )

    msg = "add_estimator does not register all estimators."
    assert len(benchmark.estimators.entities) == 2, msg


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_raise_id_restraint():
    """Test to ensure ID format is raised for malformed input ID."""
    # format of the form [username/](entity-name)-v(major).(minor)
    id_format = r"^(?:[\w:-]+\/)?([\w:.\-{}=\[\]]+)-v([\d.]+)$"
    error_msg = "Attempted to register malformed entity ID"
    benchmark = ForecastingBenchmark(id_format)
    with pytest.raises(ValueError) as exc_info:
        benchmark.add_estimator(NaiveForecaster(), "test_id")
    assert exc_info.type is ValueError, "Must raise a ValueError"
    assert error_msg in exc_info.value.args[0], "Error msg is not raised"
