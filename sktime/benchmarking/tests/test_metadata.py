"""Tests for opt-in benchmark run metadata."""

import json
import platform
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import sktime
from sktime.benchmarking.benchmarks import _json_safe, _object_tags
from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.classification.dummy import DummyClassifier
from sktime.datasets import Longley
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.panel import make_classification_problem

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _data_loader_simple() -> pd.DataFrame:
    return pd.DataFrame([2, 2, 3])


def _fast_forecasting_benchmark():
    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(NaiveForecaster(strategy="last"))
    benchmark.add_task(
        _data_loader_simple,
        ExpandingWindowSplitter(initial_window=1, step_length=1, fh=1),
        [MeanAbsoluteError()],
    )
    return benchmark


def _metadata_file(results_file):
    return results_file.with_name(f"{results_file.stem}_metadata.json")


def test_run_does_not_write_metadata_by_default(tmp_path):
    """Metadata is omitted unless return_metadata is set."""
    benchmark = _fast_forecasting_benchmark()
    results_file = tmp_path / "results.csv"

    assert benchmark.metadata is None
    results = benchmark.run(results_file)

    assert "model_id" in results.columns
    assert benchmark.metadata is None
    assert results_file.exists()
    assert not _metadata_file(results_file).exists()


def test_metadata_without_output_file_stays_in_memory():
    """return_metadata=True records metadata even when results are not saved."""
    benchmark = _fast_forecasting_benchmark()

    results = benchmark.run(None, return_metadata=True)

    assert len(results) == 1
    metadata = benchmark.metadata
    assert metadata is not None
    payload = json.loads(json.dumps(metadata.to_dict()))
    assert payload == metadata.to_dict()
    task = payload["tasks"][0]
    assert task["dataset"]["id"] == "_data_loader_simple"
    assert task["dataset"]["tags"] is None
    assert task["cv_splitter"]["tags"]["split_type"] == "temporal"


def test_metadata_file_describes_the_registered_run(tmp_path):
    """Metadata JSON follows the registered tasks, estimators, and environment."""
    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(NaiveForecaster(strategy="last"), estimator_id="naive_last")
    benchmark.add_estimator(NaiveForecaster(strategy="mean"), estimator_id="naive_mean")
    benchmark.add_task(
        Longley(),
        ExpandingWindowSplitter(initial_window=10, step_length=6, fh=1),
        [MeanAbsoluteError()],
    )

    results_file = tmp_path / "results.csv"
    results = benchmark.run(results_file, return_metadata=True)
    metadata_file = _metadata_file(results_file)

    assert metadata_file.exists()
    assert metadata_file.name == "results_metadata.json"
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert payload == benchmark.metadata.to_dict()
    assert json.loads(json.dumps(payload)) == payload

    assert set(payload) == {
        "schema_version",
        "timestamp_utc",
        "sktime_version",
        "python_version",
        "tasks",
        "estimators",
    }
    assert payload["schema_version"] == 1
    assert TIMESTAMP_RE.match(payload["timestamp_utc"])
    assert payload["sktime_version"] == sktime.__version__
    assert payload["python_version"] == platform.python_version()
    assert "dependencies" not in payload

    assert {row["estimator_id"] for row in payload["estimators"]} == {
        "naive_last",
        "naive_mean",
    }
    assert set(results["model_id"]) == {"naive_last", "naive_mean"}
    naive_last = next(
        row for row in payload["estimators"] if row["estimator_id"] == "naive_last"
    )
    assert naive_last["class_name"] == "NaiveForecaster"
    assert naive_last["tags"]["object_type"] == "forecaster"
    assert naive_last["tags"]["capability:missing_values"] is True
    assert naive_last["tags"]["tests:core"] is True
    assert "X_inner_mtype" in naive_last["tags"]

    assert len(payload["tasks"]) == 1
    task = payload["tasks"][0]
    assert task["task_id"] == results["validation_id"].iloc[0]
    assert set(task) == {
        "task_id",
        "dataset",
        "cv_splitter",
        "scorers",
        "cv_global",
    }
    assert task["dataset"]["id"] == "longley"
    assert task["dataset"]["tags"]["name"] == "longley"
    assert task["dataset"]["tags"]["is_univariate"] is True
    assert task["dataset"]["tags"]["has_exogenous"] is True
    assert task["dataset"]["tags"]["tests:core"] is True
    assert task["cv_splitter"]["id"] == "ExpandingWindowSplitter"
    assert task["cv_splitter"]["tags"]["split_type"] == "temporal"
    assert task["scorers"][0]["id"] == "MeanAbsoluteError"
    assert task["scorers"][0]["tags"]["lower_is_better"] is True
    assert task["cv_global"] is None


def test_sklearn_components_without_tags_are_left_untagged(tmp_path):
    """sklearn splitters and metric functions are recorded without invented tags."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_task(
        make_classification_problem,
        KFold(n_splits=2),
        [accuracy_score],
    )

    results_file = tmp_path / "results.json"
    results = benchmark.run(results_file, return_metadata=True)
    payload = json.loads(_metadata_file(results_file).read_text(encoding="utf-8"))

    assert results_file.exists()
    assert results["model_id"].iloc[0] == "DummyClassifier"

    task = payload["tasks"][0]
    assert task["dataset"]["id"] == "make_classification_problem"
    assert task["dataset"]["tags"] is None
    assert task["cv_splitter"]["id"] == "KFold"
    assert task["cv_splitter"]["tags"] is None
    assert task["scorers"][0]["id"] == "accuracy_score"
    assert task["scorers"][0]["tags"] is None
    assert task["cv_global"] is None

    estimator = payload["estimators"][0]
    assert estimator["estimator_id"] == "DummyClassifier"
    assert estimator["class_name"] == "DummyClassifier"
    assert estimator["tags"]["object_type"] == "classifier"


def test_object_tags_keep_full_tag_set_or_none():
    """Tag collection keeps every sktime tag and leaves other objects untagged."""
    assert _object_tags(Longley)["name"] == "longley"
    assert _object_tags(Longley())["is_univariate"] is True
    assert _object_tags(Longley())["tests:core"] is True
    assert _object_tags(KFold) is None
    assert _object_tags(KFold(n_splits=2)) is None
    assert _object_tags(accuracy_score) is None


def test_json_safe_coerces_numpy_tag_values():
    """NumPy tag values are converted to JSON-native types."""
    value = _json_safe(
        {
            "count": np.int64(3),
            "flag": np.bool_(True),
            "scores": np.array([1.0, 2.5]),
            "index_type": pd.DatetimeIndex,
        }
    )

    assert value == {
        "count": 3,
        "flag": True,
        "scores": [1.0, 2.5],
        "index_type": "DatetimeIndex",
    }
    assert json.loads(json.dumps(value)) == value
