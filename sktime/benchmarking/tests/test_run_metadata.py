"""Tests for the benchmark run metadata sidecar."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

import sktime
from sktime.benchmarking._run_metadata import (
    METADATA_SUFFIX,
    collect_run_metadata,
    get_metadata_path,
    write_run_metadata,
)
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter

# writing a .parquet results file needs a parquet engine, the sidecar itself
# does not; the run must complete for the sidecar to be written at all
_RESULTS_NAMES = [
    "results.csv",
    "results.json",
    pytest.param(
        "r.parquet",
        marks=pytest.mark.skipif(
            not _check_soft_dependencies("pyarrow", severity="none"),
            reason="parquet results files require a parquet engine",
        ),
    ),
]

_EXPECTED_KEYS = {
    "sktime_version",
    "timestamp",
    "n_tasks",
    "n_estimators",
    "output_path",
    "sys_info",
    "deps_info",
}


def _data_loader_simple() -> pd.DataFrame:
    return pd.DataFrame([2, 2, 3])


@pytest.fixture
def cv_splitter():
    return ExpandingWindowSplitter(initial_window=1, step_length=1, fh=1)


@pytest.fixture
def benchmark(cv_splitter):
    """Benchmark with two estimators and one task."""
    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(NaiveForecaster(strategy="mean"), estimator_id="mean")
    benchmark.add_estimator(NaiveForecaster(strategy="last"), estimator_id="last")
    benchmark.add_task(_data_loader_simple, cv_splitter, [MeanAbsoluteError()])
    return benchmark


@pytest.mark.parametrize("results_name", ["results.csv", "results.json", "r.parquet"])
def test_metadata_path_appends_suffix(results_name):
    """Sidecar path appends .meta.json to the full results path."""
    assert get_metadata_path(results_name) == Path(results_name + METADATA_SUFFIX)


def test_metadata_path_is_injective_across_formats():
    """Results files differing only by extension get distinct sidecars."""
    paths = {get_metadata_path(f"results.{ext}") for ext in ("csv", "json", "parquet")}
    assert len(paths) == 3


def test_metadata_path_accepts_path_objects(tmp_path):
    """Sidecar path derivation works for pathlib.Path input."""
    results_path = tmp_path / "results.csv"
    assert get_metadata_path(results_path) == Path(str(results_path) + METADATA_SUFFIX)


def test_metadata_path_none_when_no_results_path():
    """No results path means no sidecar path."""
    assert get_metadata_path(None) is None


def test_collect_run_metadata_fields():
    """Collected metadata has the expected keys and values."""
    metadata = collect_run_metadata("results.csv", n_tasks=2, n_estimators=3)

    assert set(metadata) == _EXPECTED_KEYS
    assert metadata["sktime_version"] == sktime.__version__
    assert metadata["n_tasks"] == 2
    assert metadata["n_estimators"] == 3
    assert metadata["output_path"] == "results.csv"
    assert set(metadata["sys_info"]) == {"python", "executable", "machine"}
    assert "sktime" in metadata["deps_info"]


def test_collect_run_metadata_timestamp_is_utc():
    """Timestamp parses as an ISO 8601 UTC timestamp."""
    before = datetime.now(timezone.utc)
    metadata = collect_run_metadata("results.csv", n_tasks=1, n_estimators=1)
    after = datetime.now(timezone.utc)

    timestamp = datetime.fromisoformat(metadata["timestamp"])
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timezone.utc.utcoffset(None)
    assert before <= timestamp <= after


def test_collect_run_metadata_is_json_serializable():
    """Metadata round-trips through JSON unchanged."""
    metadata = collect_run_metadata("results.csv", n_tasks=1, n_estimators=1)
    assert json.loads(json.dumps(metadata)) == metadata


def test_write_run_metadata_writes_sidecar(tmp_path):
    """Sidecar is written at the derived path with the collected metadata."""
    results_path = tmp_path / "results.csv"

    metadata_path = write_run_metadata(results_path, n_tasks=2, n_estimators=3)

    assert metadata_path == Path(str(results_path) + METADATA_SUFFIX)
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert set(metadata) == _EXPECTED_KEYS
    assert metadata["n_tasks"] == 2
    assert metadata["n_estimators"] == 3
    assert metadata["output_path"] == str(results_path)


def test_write_run_metadata_none_path_writes_nothing(tmp_path):
    """No results path means no sidecar is written."""
    assert write_run_metadata(None, n_tasks=1, n_estimators=1) is None
    assert list(tmp_path.iterdir()) == []


def test_write_run_metadata_overwrites_existing(tmp_path):
    """A second run overwrites the sidecar rather than appending to it."""
    results_path = tmp_path / "results.csv"

    write_run_metadata(results_path, n_tasks=1, n_estimators=1)
    metadata_path = write_run_metadata(results_path, n_tasks=5, n_estimators=7)

    metadata = json.loads(metadata_path.read_text())
    assert metadata["n_tasks"] == 5
    assert metadata["n_estimators"] == 7


def test_write_run_metadata_leaves_no_temp_file(tmp_path):
    """The atomic write leaves no temporary file behind."""
    results_path = tmp_path / "results.csv"

    write_run_metadata(results_path, n_tasks=1, n_estimators=1)

    assert [p.name for p in tmp_path.iterdir()] == ["results.csv" + METADATA_SUFFIX]


@pytest.mark.parametrize("results_name", _RESULTS_NAMES)
def test_run_writes_sidecar_for_all_backends(benchmark, tmp_path, results_name):
    """benchmark.run writes a sidecar next to the results file, in any format."""
    results_path = tmp_path / results_name

    benchmark.run(results_path)

    metadata_path = Path(str(results_path) + METADATA_SUFFIX)
    assert results_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["n_tasks"] == 1
    assert metadata["n_estimators"] == 2
    assert metadata["output_path"] == str(results_path)
    assert metadata["sktime_version"] == sktime.__version__


def test_run_warns_and_continues_when_sidecar_write_fails(
    benchmark, tmp_path, monkeypatch
):
    """A failed sidecar write warns, and leaves the run and its results intact."""

    def _raise_oserror(path, contents):
        raise OSError("No space left on device")

    monkeypatch.setattr(
        "sktime.benchmarking._run_metadata._atomic_write_text", _raise_oserror
    )
    results_path = tmp_path / "results.csv"

    with pytest.warns(UserWarning, match="Failed to write benchmark run metadata"):
        results_df = benchmark.run(results_path)

    assert len(results_df) == 2
    assert results_path.exists()
    assert not Path(str(results_path) + METADATA_SUFFIX).exists()


def test_write_run_metadata_returns_none_when_write_fails(tmp_path, monkeypatch):
    """A failed sidecar write reports that nothing was written."""

    def _raise_oserror(path, contents):
        raise OSError("Permission denied")

    monkeypatch.setattr(
        "sktime.benchmarking._run_metadata._atomic_write_text", _raise_oserror
    )

    with pytest.warns(UserWarning, match="Permission denied"):
        metadata_path = write_run_metadata(
            tmp_path / "results.csv", n_tasks=1, n_estimators=1
        )

    assert metadata_path is None


def test_run_without_output_file_writes_no_sidecar(benchmark, tmp_path, monkeypatch):
    """A run that persists nothing writes no sidecar."""
    monkeypatch.chdir(tmp_path)

    results_df = benchmark.run()

    assert not results_df.empty
    assert list(tmp_path.iterdir()) == []


def test_run_does_not_change_results_file_contents(benchmark, tmp_path):
    """Metadata stays out of the results file itself."""
    results_path = tmp_path / "results.json"

    benchmark.run(results_path)

    results = json.loads(results_path.read_text())
    assert isinstance(results, list)
    assert len(results) == 2
    for row in results:
        assert not _EXPECTED_KEYS.intersection(row)
