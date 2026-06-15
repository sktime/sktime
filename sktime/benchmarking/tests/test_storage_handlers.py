import pandas as pd
import pytest

from sktime.benchmarking._benchmarking_dataclasses import FoldResults, ResultObject
from sktime.benchmarking._storage_handlers import (
    CSVStorageHandler,
    JSONStorageHandler,
    PickleStorageHandler,
    # ParquetStorageHandler,
)

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
    "storage_handler,file_extension",
    [
        (JSONStorageHandler, ".json"),
        (CSVStorageHandler, ".csv"),
        (PickleStorageHandler, ".pkl"),
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
        (PickleStorageHandler, ".pkl"),
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


def test_pickle_preserves_non_default_index(tmp_path):
    """Test the pickle handler round-trips returned data with a non-default index.

    The tabular backends (parquet, csv) serialize the returned ``ground_truth``,
    ``predictions`` and ``train_data`` frames via an intermediate dict
    representation that does not handle ``pandas`` index types such as
    ``PeriodIndex`` (the index used by ``load_airline`` and many other sktime
    datasets). The pickle backend serializes the result objects directly and
    therefore preserves the index losslessly.
    """
    index = pd.period_range("2000-01", periods=3, freq="M")
    ground_truth = pd.DataFrame({"data": [1.0, 0.0, 1.0]}, index=index)
    predictions = pd.DataFrame({"data": [1.0, 0.0, 1.0]}, index=index)
    train_data = pd.DataFrame({"data": [0.0, 1.0, 0.0]}, index=index)

    sample_results = [
        ResultObject(
            model_id="model_1",
            task_id="val_1",
            folds={
                0: FoldResults(
                    scores={"accuracy": 0.9},
                    ground_truth=ground_truth,
                    predictions=predictions,
                    train_data=train_data,
                )
            },
        )
    ]

    handler = PickleStorageHandler(tmp_path / "results.pkl")
    handler.save(sample_results)
    results = handler.load()

    assert len(results) == 1
    pd.testing.assert_frame_equal(results[0].folds[0].ground_truth, ground_truth)
    pd.testing.assert_frame_equal(results[0].folds[0].predictions, predictions)
    pd.testing.assert_frame_equal(results[0].folds[0].train_data, train_data)
