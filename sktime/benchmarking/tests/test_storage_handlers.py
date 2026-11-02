import pandas as pd
import pytest

from sktime.benchmarking._benchmarking_dataclasses import FoldResults, ResultObject
from sktime.benchmarking._storage_handlers import (
    CSVStorageHandler,
    JSONStorageHandler,
    _parse_frame_literal,
    # ParquetStorageHandler,
)
from sktime.benchmarking.benchmarks import BenchmarkingResults

RESULT_OBJECT_LISTS = [
    [
        ResultObject(
            model_id="model_1",
            task_id="val_1",
            folds={
                0: FoldResults(
                    scores={"accuracy": 0.9, "f1": 0.8},
                    ground_truth=pd.DataFrame({"data": [1.0, 0.0, 1.0]}),
                    predictions=pd.DataFrame({"data": [1.0, 0.0, 1.0]}),
                    train_data=pd.DataFrame({"data": [0.0, 1.0, 0.0]}),
                )
            },
        )
    ],
    [
        ResultObject(
            model_id="model_1",
            task_id="val_1",
            folds={
                0: FoldResults(
                    scores={
                        "accuracy": pd.Series([0.8, 0.9], name="accuracy"),
                        "f1": 0.8,
                    },
                    ground_truth=pd.DataFrame({"data": [1.0, 0.0, 1.0]}),
                    predictions=pd.DataFrame({"data": [1.0, 0.0, 1.0]}),
                    train_data=pd.DataFrame({"data": [0.0, 1.0, 0.0]}),
                )
            },
        )
    ],
]


@pytest.mark.parametrize(
    "index",
    [
        pd.period_range("2020-01", periods=3, freq="M", name="time"),
        pd.date_range("2020-01-01", periods=3, name="time"),
        pd.date_range("2020-01-01", periods=3, tz="UTC", name="time"),
        pd.MultiIndex.from_product(
            [["series"], pd.period_range("2020-01", periods=3, freq="M")],
            names=["instance", "time"],
        ),
    ],
)
def test_csv_roundtrip_time_index(tmp_path, index):
    """CSV results retain pandas time indexes in every stored fold frame."""
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=index)
    result = ResultObject(
        model_id="model",
        task_id="task",
        folds={
            0: FoldResults(
                scores={"error": 0.0},
                ground_truth=frame,
                predictions=frame + 1,
                train_data=frame + 2,
            )
        },
    )
    handler = CSVStorageHandler(tmp_path / "results.csv")
    handler.save([result])

    loaded = handler.load()[0].folds[0]
    for name in ["ground_truth", "predictions", "train_data"]:
        pd.testing.assert_frame_equal(
            getattr(loaded, name), getattr(result.folds[0], name), check_freq=False
        )


@pytest.mark.parametrize(
    "value",
    [
        "[open('results.csv')]",
        "[__import__('os')]",
        "[Timestamp(__import__('os'))]",
        "[pd.Timestamp('2020-01-01')]",
        "[Timestamp('2020-01-01').date()]",
        "[Period(**{'value': '2020-01', 'freq': 'M'})]",
    ],
)
def test_csv_frame_literal_rejects_other_calls(value):
    """The CSV parser accepts time literals without evaluating arbitrary calls."""
    with pytest.raises(ValueError, match="malformed node or string"):
        _parse_frame_literal(value)


@pytest.mark.parametrize(
    "storage_handler,file_extension",
    [
        (JSONStorageHandler, ".json"),
        (CSVStorageHandler, ".csv"),
        # (ParquetStorageHandler, ".parquet"),
    ],
)
@pytest.mark.parametrize("sample_results", RESULT_OBJECT_LISTS)
def test_store_load_results(tmp_path, storage_handler, file_extension, sample_results):
    handler = storage_handler(tmp_path / f"results{file_extension}")

    handler.save(sample_results)
    results = handler.load()

    assert len(results) == 1
    assert results[0].model_id == sample_results[0].model_id
    assert results[0].task_id == sample_results[0].task_id
    if isinstance(sample_results[0].folds[0].scores["accuracy"], pd.DataFrame):
        pd.testing.assert_frame_equal(
            results[0].folds[0].scores["accuracy"],
            sample_results[0].folds[0].scores["accuracy"],
        )
    else:
        assert (
            results[0].folds[0].scores["accuracy"]
            == sample_results[0].folds[0].scores["accuracy"]
        )
    assert results[0].folds[0].scores["f1"] == sample_results[0].folds[0].scores["f1"]
    if file_extension in [".csv"]:
        # CSV does not support storing ground_truth, predictions, and train_data
        return

    pd.testing.assert_frame_equal(
        results[0].folds[0].ground_truth, sample_results[0].folds[0].ground_truth
    )
    pd.testing.assert_frame_equal(
        results[0].folds[0].predictions, sample_results[0].folds[0].predictions
    )
    pd.testing.assert_frame_equal(
        results[0].folds[0].train_data, sample_results[0].folds[0].train_data
    )


@pytest.mark.parametrize(
    "storage_handler,file_extension",
    [
        (JSONStorageHandler, ".json"),
        (CSVStorageHandler, ".csv"),
        # (ParquetStorageHandler, ".parquet"),
    ],
)
def test_store_load_results_empty_training(tmp_path, storage_handler, file_extension):
    handler = storage_handler(tmp_path / f"results{file_extension}")

    handler.save(
        [
            ResultObject(
                model_id="model_1",
                task_id="val_1",
                folds={
                    0: FoldResults(
                        scores={"f1": 0.8},
                        ground_truth=None,
                        predictions=None,
                        train_data=None,
                    )
                },
            )
        ]
    )

    results = handler.load()

    assert len(results) == 1
    assert results[0].model_id == "model_1"
    assert results[0].task_id == "val_1"

    assert results[0].folds[0].scores["f1"] == 0.8

    assert results[0].folds[0].ground_truth is None
    assert results[0].folds[0].predictions is None
    assert results[0].folds[0].train_data is None


@pytest.mark.parametrize(
    "storage_handler,file_extension",
    [
        (JSONStorageHandler, ".json"),
        (CSVStorageHandler, ".csv"),
    ],
)
@pytest.mark.parametrize("sample_results", RESULT_OBJECT_LISTS)
def test_benchmarking_results_to_df(
    tmp_path, storage_handler, file_extension, sample_results
):
    path = tmp_path / f"results{file_extension}"
    storage_handler(path).save(sample_results)

    loaded_df = BenchmarkingResults(path=str(path)).to_df()
    expected = BenchmarkingResults.__new__(BenchmarkingResults)
    expected.results = sample_results
    expected_df = expected.to_df()

    pd.testing.assert_frame_equal(loaded_df, expected_df)


def test_benchmarking_results_to_df_missing_file(tmp_path):
    path = tmp_path / "results.csv"
    loaded_df = BenchmarkingResults(path=str(path)).to_df()
    assert loaded_df.empty


def test_benchmarking_results_unsupported_extension(tmp_path):
    path = tmp_path / "results.txt"
    path.touch()
    with pytest.raises(ValueError, match="No storage handler found"):
        BenchmarkingResults(path=str(path))
