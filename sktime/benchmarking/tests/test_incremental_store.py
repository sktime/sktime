"""Tests for incremental benchmark result persistence."""

import hashlib
import json

import pandas as pd
import pytest

from sktime.benchmarking._benchmarking_dataclasses import FoldResults, ResultObject
from sktime.benchmarking._incremental_store import (
    IncrementalResultStore,
    result_key,
)
from sktime.benchmarking._storage_handlers import (
    CSVStorageHandler,
    JSONStorageHandler,
)
from sktime.benchmarking.benchmarks import _BenchmarkingResults, _is_failed_result
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanSquaredPercentageError
from sktime.split import ExpandingWindowSplitter
from sktime.tests.test_switch import run_test_module_changed


def _sample_result(task_id="val_1", model_id="model_1", accuracy=0.9):
    return ResultObject(
        model_id=model_id,
        task_id=task_id,
        folds={
            0: FoldResults(
                scores={"accuracy": accuracy, "f1": 0.8},
                ground_truth=pd.DataFrame({"data": [1.0, 0.0, 1.0]}),
                predictions=pd.DataFrame({"data": [1.0, 0.0, 1.0]}),
                train_data=pd.DataFrame({"data": [0.0, 1.0, 0.0]}),
            )
        },
    )


@pytest.mark.parametrize("output_suffix", [".json", ".csv", ".parquet"])
def test_incremental_store_save_and_load(tmp_path, output_suffix):
    """Test saving and reloading a result through the incremental store."""
    output_file = tmp_path / f"results{output_suffix}"
    store = IncrementalResultStore(output_file)
    result = _sample_result()

    store.save_result(result)
    loaded = store.load_results()

    assert len(loaded) == 1
    assert loaded[0].task_id == result.task_id
    assert loaded[0].model_id == result.model_id
    assert loaded[0].folds[0].scores["accuracy"] == result.folds[0].scores["accuracy"]


def test_result_key_is_stable():
    """Test deterministic key generation for the same task-model pair."""
    assert result_key("task_a", "model_b") == result_key("task_a", "model_b")
    assert result_key("task_a", "model_b") != result_key("task_b", "model_a")


def test_incremental_store_ignores_missing_done_file(tmp_path):
    """Test skipping partial results without a completion marker."""
    output_file = tmp_path / "results.json"
    store = IncrementalResultStore(output_file)
    result = _sample_result()
    store.parts_dir.mkdir(parents=True)

    key = result_key(result.task_id, result.model_id)
    json_path = store.parts_dir / f"{key}.json"
    json_path.write_text(json.dumps(JSONStorageHandler.serialize_result(result)))

    assert store.load_results() == []


def test_incremental_store_ignores_hash_mismatch(tmp_path):
    """Test skipping results whose checksum does not match the stored hash."""
    output_file = tmp_path / "results.json"
    store = IncrementalResultStore(output_file)
    result = _sample_result()
    store.save_result(result)

    key = result_key(result.task_id, result.model_id)
    done_path = store.parts_dir / f"{key}.done"
    done_path.write_text(json.dumps({"sha256": "invalid_hash"}))

    assert store.load_results() == []


def test_incremental_store_ignores_invalid_json(tmp_path):
    """Test skipping corrupted JSON result files."""
    output_file = tmp_path / "results.json"
    store = IncrementalResultStore(output_file)
    result = _sample_result()
    store.save_result(result)

    key = result_key(result.task_id, result.model_id)
    json_path = store.parts_dir / f"{key}.json"
    json_path.write_text("{not valid json")

    assert store.load_results() == []


def test_incremental_store_disabled_when_no_output_path():
    """Test disabling incremental persistence when no output path is provided."""
    store = IncrementalResultStore(None)
    result = _sample_result()

    store.save_result(result)

    assert store.parts_dir is None
    assert store.load_results() == []


def test_done_file_contains_checksum(tmp_path):
    """Test writing checksum metadata in the done file."""
    output_file = tmp_path / "results.json"
    store = IncrementalResultStore(output_file)
    result = _sample_result()
    store.save_result(result)

    key = result_key(result.task_id, result.model_id)
    json_path = store.parts_dir / f"{key}.json"
    done_path = store.parts_dir / f"{key}.done"

    contents = json_path.read_text(encoding="utf-8")
    expected_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    done_data = json.loads(done_path.read_text(encoding="utf-8"))

    assert done_data == {"sha256": expected_hash}


def test_benchmarking_results_loads_from_parts_on_resume(tmp_path):
    """Test loading previously persisted partial results during resume."""
    output_file = tmp_path / "results.json"
    store = IncrementalResultStore(output_file)
    store.save_result(_sample_result(task_id="val_1", model_id="model_1"))
    store.save_result(_sample_result(task_id="val_2", model_id="model_2"))

    results = _BenchmarkingResults(path=str(output_file))

    assert len(results.results) == 2
    assert results.contains("val_1", "model_1")
    assert results.contains("val_2", "model_2")


def test_benchmarking_results_prefers_final_output_file(tmp_path):
    """Test preferring the finalized results file over incremental artifacts."""
    output_file = tmp_path / "results.json"
    store = IncrementalResultStore(output_file)
    store.save_result(_sample_result(task_id="val_1", model_id="model_1", accuracy=0.5))

    final_result = _sample_result(task_id="val_1", model_id="model_1", accuracy=0.99)
    JSONStorageHandler(output_file).save([final_result])

    results = _BenchmarkingResults(path=str(output_file))

    assert len(results.results) == 1
    assert results.results[0].folds[0].scores["accuracy"] == 0.99


@pytest.mark.parametrize(
    "storage_handler,file_extension",
    [
        (JSONStorageHandler, ".json"),
        (CSVStorageHandler, ".csv"),
    ],
)
def test_benchmarking_results_incremental_save_and_final_merge(
    tmp_path, storage_handler, file_extension
):
    """Test merging incremental artifacts into the final output file."""
    output_file = tmp_path / f"results{file_extension}"
    results = _BenchmarkingResults(path=str(output_file))

    results.update(_sample_result(task_id="val_1", model_id="model_1"))
    results.update(_sample_result(task_id="val_2", model_id="model_2"))

    parts_dir = IncrementalResultStore(output_file).parts_dir
    assert parts_dir.exists()
    assert len(list(parts_dir.glob("*.json"))) == 2

    results.save()

    assert output_file.exists()
    assert not IncrementalResultStore(output_file).parts_dir.exists()

    loaded = storage_handler(output_file).load()
    assert len(loaded) == 2


def test_benchmarking_results_update_replaces_existing_result(tmp_path):
    """Test replacing an existing result when the same task-model pair is updated."""
    output_file = tmp_path / "results.json"
    results = _BenchmarkingResults(path=str(output_file))

    results.update(_sample_result(task_id="val_1", model_id="model_1", accuracy=0.5))
    results.update(_sample_result(task_id="val_1", model_id="model_1", accuracy=0.99))

    assert len(results.results) == 1
    assert results.results[0].folds[0].scores["accuracy"] == 0.99

    loaded = IncrementalResultStore(output_file).load_results()
    assert len(loaded) == 1
    assert loaded[0].folds[0].scores["accuracy"] == 0.99


def _data_loader_simple() -> pd.DataFrame:
    """Return simple data for use in testing."""
    return pd.DataFrame([2, 2, 3])


def _setup_benchmark():
    """Create an experiment with two estimators and one task."""
    benchmark = ForecastingBenchmark()

    benchmark.add_estimator(
        NaiveForecaster(strategy="last"),
        estimator_id="naive_last",
    )
    benchmark.add_estimator(
        NaiveForecaster(strategy="mean"),
        estimator_id="naive_mean",
    )

    cv_splitter = ExpandingWindowSplitter(
        initial_window=1,
        step_length=1,
        fh=1,
    )
    benchmark.add_task(
        _data_loader_simple,
        cv_splitter,
        [MeanSquaredPercentageError()],
    )

    return benchmark


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("output_suffix", [".json", ".csv"])
def test_results_saved_after_each_step(tmp_path, monkeypatch, output_suffix):
    """Test persisting each completed experiment before the next one starts."""
    benchmark = _setup_benchmark()
    results_path = tmp_path / f"results{output_suffix}"

    original_update = _BenchmarkingResults.update
    n_completed = 0

    def counting_update(self, new_result):
        nonlocal n_completed
        original_update(self, new_result)
        n_completed += 1

        store = IncrementalResultStore(results_path)
        assert store.parts_dir.exists()
        assert len(store.load_results()) == n_completed
        assert not results_path.exists()

    monkeypatch.setattr(_BenchmarkingResults, "update", counting_update)

    results_df = benchmark.run(str(results_path))

    assert results_path.exists()
    assert len(results_df) == 2
    assert not IncrementalResultStore(results_path).parts_dir.exists()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("output_suffix", [".json", ".csv"])
def test_benchmark_resumes_after_crash(tmp_path, monkeypatch, output_suffix):
    """Test resuming a benchmark after a crash while preserving completed experiments."""  # NOQA: E501
    results_path = tmp_path / f"results{output_suffix}"

    benchmark = _setup_benchmark()
    original_run_validation = benchmark._run_validation
    n_calls = 0

    def crashing_run_validation(task, estimator):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 2:
            raise RuntimeError("Simulated benchmark failure")
        return original_run_validation(task, estimator)

    monkeypatch.setattr(benchmark, "_run_validation", crashing_run_validation)

    original_save = _BenchmarkingResults.save
    save_calls = 0

    def crashing_save_once(self):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            raise RuntimeError("Simulated benchmark failure")
        return original_save(self)

    # Failed validation runs are checkpointed with NaN; simulate a crash during
    # final save instead.
    monkeypatch.setattr(_BenchmarkingResults, "save", crashing_save_once)

    with pytest.raises(RuntimeError, match="Simulated benchmark failure"):
        benchmark.run(str(results_path))

    store = IncrementalResultStore(results_path)
    partial_results = store.load_results()

    assert not results_path.exists()
    assert store.parts_dir.exists()
    assert len(partial_results) == 2
    successful = [r for r in partial_results if not _is_failed_result(r)]
    failed = [r for r in partial_results if _is_failed_result(r)]
    assert len(successful) == 1
    assert successful[0].model_id == "naive_last"
    assert len(failed) == 1
    assert failed[0].model_id == "naive_mean"

    resumed_benchmark = _setup_benchmark()
    results_df = resumed_benchmark.run(str(results_path))

    assert results_path.exists()
    assert len(results_df) == 2
    assert set(results_df["model_id"]) == {"naive_last", "naive_mean"}
    assert not store.parts_dir.exists()
