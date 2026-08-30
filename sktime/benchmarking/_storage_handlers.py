"""Final-format storage handlers for benchmark results.

Each `BaseStorageHandler` subclass implements the strategy for a
specific output file format (JSON, CSV, Parquet). The appropriate handler is
selected by file extension via `get_storage_backend`.

Crash-safe incremental checkpoints during a run are not handled here,
they live in `sktime.benchmarking._incremental_store` and are coordinated
by `sktime.benchmarking._results_persistence`.
"""

import abc
import ast
import json
import pickle
from pathlib import Path

import pandas as pd

from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    ResultObject,
    asdict,
)


# Atomic writes to ensure old result file isn't deleted before
# new file is saved
def _atomic_write_text(path: Path, contents: str) -> None:
    """Write text to a file atomically via a temporary file.

    Parameters
    ----------
    path : Path
        Destination file path. Must refer to a file, not a directory.
    contents : str
        Text to write to the file.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(contents)
    tmp_path.replace(path)


def _atomic_write_path(path: Path, write_fn) -> None:
    """Write to *path* atomically via a temporary file.

    Parameters
    ----------
    path : Path
        Destination file path. Must refer to a file, not a directory.
    write_fn : callable
        Callable accepting the temporary path and writing the file contents.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp_path)
    tmp_path.replace(path)


class BaseStorageHandler(abc.ABC):
    """Handles storage of benchmark results to and from a single file.

    Each subclass implements read/write for one output format (JSON, CSV,
    or Parquet). The file extension of ``path`` determines which handler
    is selected via `get_storage_backend`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the results file. Must refer to a file, not a directory.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path

    @abc.abstractmethod
    def save(self, results: ResultObject):
        """Save benchmark results to the file at ``self.path``.

        Overwrites the existing file if present. Writes are atomic.

        Parameters
        ----------
        results : list of ResultObject
            Benchmark results to persist.
        """

    def load(self) -> list[ResultObject]:
        """Load benchmark results from the file at ``self.path``.

        Returns
        -------
        list of ResultObject
            Loaded results. Returns an empty list if the file does not exist.
        """
        if self.path is not None and not Path(self.path).exists():
            return []
        return self._load()

    @abc.abstractmethod
    def _load(self) -> list[ResultObject]:
        """Load benchmark results from an existing file at ``self.path``.

        Assumes the file exists; called only by `load` after that
        check. Subclasses implement format-specific parsing here.

        Returns
        -------
        list of ResultObject
            Loaded results.
        """

    @staticmethod
    @abc.abstractmethod
    def is_applicable(path):
        """Return whether this handler supports the given results file path.

        Parameters
        ----------
        path : pathlib.Path or None
            Path to the results file, or ``None`` for
            `NullStorageHandler`. Must refer to a file, not a
            directory, when not ``None``.

        Returns
        -------
        bool
            ``True`` if this handler should be used for ``path``.
        """


class JSONStorageHandler(BaseStorageHandler):
    """Storage handler for JSON files, with ending .json.

    Loads and saves results in JSON format, as follows.

    Each result is stored as a JSON record with the following keys:

    - ``model_id``: The ID of the model.
    - ``validation_id``: The ID of the validation run.
    - ``folds``: A dictionary of fold results,
        where each key is a fold ID and the value
        is a dictionary of scores and dataframes.

    The ``folds`` dictionary has the following structure,
    where ``ground_truth``, ``predictions``, and ``train_data``
    are returned only if they were requested during benchmarking.

    Keys are strings representing fold IDs.

    ```json
    {
        "0": {
            "scores": {
                "accuracy": 0.9,
                "f1_score": 0.8
            },
            "ground_truth": {
                "column1": [1, 0, 1],
                "column2": [0.5, 0.3, 0.8]
            },
            "predictions": {
                "column1": [1, 0, 0],
                "column2": [0.6, 0.4, 0.7]
            },
            "train_data": {
                "column1": [0, 1, 1],
                "column2": [0.2, 0.9, 0.4]
            }
        },
        "1": {
            ...
        }
    }

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JSON results file. Must refer to a file with a
        ``.json`` extension, not a directory.
    """

    @staticmethod
    def serialize_result(result: ResultObject) -> dict:
        """Serialize a ResultObject to the JSON storage format."""
        return asdict(result, pd_orient="list")

    @staticmethod
    def deserialize_result(row: dict) -> ResultObject:
        """Deserialize a ResultObject from the JSON storage format."""
        folds = {}
        for fold_id, fold in row["folds"].items():
            scores = {}
            for score_name, score_val in fold["scores"].items():
                if isinstance(score_val, dict):
                    score_val = pd.DataFrame.from_records(score_val)
                scores[score_name] = score_val
            if "ground_truth" in fold and fold["ground_truth"]:
                ground_truth = pd.DataFrame(fold["ground_truth"])
            else:
                ground_truth = None
            if "predictions" in fold and fold["predictions"]:
                predictions = pd.DataFrame(fold["predictions"])
            else:
                predictions = None
            if "train_data" in fold and fold["train_data"]:
                train_data = pd.DataFrame(fold["train_data"])
            else:
                train_data = None
            folds[int(fold_id)] = FoldResults(
                scores, ground_truth, predictions, train_data
            )

        return ResultObject(
            model_id=row["model_id"],
            task_id=row["validation_id"],
            folds=folds,
        )

    def save(self, results: list[ResultObject]):
        """Save benchmark results to the JSON file at ``self.path``.

        Parameters
        ----------
        results : list of ResultObject
            Benchmark results to persist as a JSON array.
        """
        path = Path(self.path)
        contents = json.dumps([self.serialize_result(x) for x in results])
        _atomic_write_text(path, contents)

    def _load(self) -> list[ResultObject]:
        """Load benchmark results from the JSON file at ``self.path``.

        Returns
        -------
        list of ResultObject
            Loaded results parsed from a JSON array of records.
        """
        with open(self.path) as f:
            rows = json.load(f)

        return [self.deserialize_result(row) for row in rows]

    @staticmethod
    def is_applicable(path):
        return path.suffix == ".json"


class ParquetStorageHandler(BaseStorageHandler):
    """Storage handler for Parquet files, with ending .parquet.

    Loads and saves results in Parquet format, as follows.
    Each result is stored as a row in a Parquet file with the following columns:

    - ``model_id``: The ID of the model.
    - ``validation_id``: The ID of the validation run.
    - ``folds``: A dictionary of fold results,
        where each key is a fold ID and the value
        is a dictionary of scores and dataframes.

    The results are stored in a tabular format,
    where each row corresponds to a single model-validation pair.

    Columns are the following:

    - ``model_id``: The ID of the model.
    - ``validation_id``: The ID of the validation run.
    - ``folds.{fold_id}.scores.{score_name}``: The score value for the given fold and
      score name.
    - ``folds.{fold_id}.ground_truth``: The ground truth dataframe for the given fold.
    - ``folds.{fold_id}.predictions``: The predictions dataframe for the given fold.
    - ``folds.{fold_id}.train_data``: The training data dataframe for the given fold.

    Columns ``ground_truth``, ``predictions``, and ``train_data``
    are included only if they were requested during benchmarking.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Parquet results file. Must refer to a file with a
        ``.parquet`` extension, not a directory.
    """

    def save(self, results: list[ResultObject]):
        """Save benchmark results to the Parquet file at ``self.path``.

        Parameters
        ----------
        results : list of ResultObject
            Benchmark results to persist as one row per model-validation pair.
        """
        if not results:
            pd.DataFrame(columns=["validation_id", "model_id"]).to_parquet(
                self.path, index=False
            )
            return

        results_df = pd.json_normalize(
            list(map(lambda x: asdict(x, pd_orient="tight"), results))
        )

        results_df = results_df.sort_values(by=["validation_id", "model_id"])

        results_df = results_df.reset_index(drop=True)
        _atomic_write_path(
            Path(self.path),
            lambda tmp_path: results_df.to_parquet(tmp_path, index=False),
        )

    def _load(self) -> list[ResultObject]:
        """Load benchmark results from the Parquet file at ``self.path``.

        Returns
        -------
        list of ResultObject
            Loaded results reconstructed from tabular rows.
        """
        results_df = pd.read_parquet(self.path)
        results = []
        for ix, row in results_df.iterrows():
            folds = _get_folds(row)

            results.append(
                ResultObject(
                    model_id=row["model_id"],
                    task_id=row["validation_id"],
                    folds=folds,
                )
            )
        return results

    @staticmethod
    def is_applicable(path):
        return path.suffix == ".parquet"


class CSVStorageHandler(BaseStorageHandler):
    """Storage handler for CSV files, with ending .csv.

    Loads and saves results in CSV format with separator comma, as follows.

    The results are stored in a tabular format,
    where each row corresponds to a single model-validation pair.

    Each result is stored as a row in a CSV file with the following columns:

    - ``model_id``: The ID of the model.
    - ``validation_id``: The ID of the validation run.
    - ``folds``: A dictionary of fold results,
        where each key is a fold ID and the value
        is a dictionary of scores and dataframes.

    The following columns are included for each fold:

    - ``folds.{fold_id}.scores.{score_name}``: The score value for the given fold and
        score name.
    - ``folds.{fold_id}.ground_truth``: The ground truth dataframe for the given fold.
    - ``folds.{fold_id}.predictions``: The predictions dataframe for the given fold.
    - ``folds.{fold_id}.train_data``: The training data dataframe for the
        given fold.

    Columns ``ground_truth``, ``predictions``, and ``train_data``
    are included only if they were requested during benchmarking.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV results file. Must refer to a file with a
        ``.csv`` extension, not a directory.
    """

    def save(self, results: list[ResultObject]):
        """Save benchmark results to the CSV file at ``self.path``.

        Parameters
        ----------
        results : list of ResultObject
            Benchmark results to persist as one row per model-validation pair.
        """
        if not results:
            pd.DataFrame(columns=["validation_id", "model_id"]).to_csv(
                self.path, index=False
            )
            return

        results_df = pd.json_normalize(
            list(map(lambda x: asdict(x, pd_orient="tight"), results))
        )

        results_df = results_df.sort_values(by=["validation_id", "model_id"])

        results_df = results_df.reset_index(drop=True)
        _atomic_write_path(
            Path(self.path), lambda tmp_path: results_df.to_csv(tmp_path, index=False)
        )

    def _load(self) -> list[ResultObject]:
        """Load benchmark results from the CSV file at ``self.path``.

        Returns
        -------
        list of ResultObject
            Loaded results reconstructed from tabular rows.
        """
        results_df = pd.read_csv(self.path)
        results = []
        for ix, row in results_df.iterrows():
            folds = _get_folds(row)
            results.append(
                ResultObject(
                    model_id=row["model_id"],
                    task_id=row["validation_id"],
                    folds=folds,
                )
            )
        return results

    @staticmethod
    def is_applicable(path):
        return path.suffix == ".csv"


class PickleStorageHandler(BaseStorageHandler):
    """Storage handler for pickle files, with ending .pkl or .pickle.

    Loads and saves results in Python's binary pickle format.

    Unlike the JSON, Parquet and CSV handlers, this handler serializes the
    ``ResultObject`` instances directly, without an intermediate conversion to a
    tabular or record-oriented representation. As a consequence:

    - The full result objects, including ``ground_truth``, ``predictions`` and
      ``train_data`` (returned when ``return_data=True``), are stored and
      restored losslessly, regardless of the index type of the underlying
      ``pandas`` objects (e.g. ``PeriodIndex``, ``DatetimeIndex``,
      ``MultiIndex``). This is the recommended backend when ``return_data=True``,
      as the tabular backends can lose information or fail to serialize such
      indices.
    - The on-disk representation is a compact binary blob, which is typically
      much smaller than the equivalent CSV or JSON file. This makes it a good
      choice for benchmarking runs over many models and datasets with
      ``return_data=True``.

    Warning
    -------
    The pickle format is not secure. Only load result files that you have
    generated yourself or that come from a trusted source, as loading a
    maliciously crafted pickle file can execute arbitrary code.

    Parameters
    ----------
    path : str, or Path coercible
        The path to the file to save to or load
    """

    def save(self, results: list[ResultObject]):
        """Save the results to a pickle file.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """
        with open(self.path, "wb") as f:
            pickle.dump(results, f)

    def _load(self) -> list[ResultObject]:
        """Load the results from a pickle file.

        Returns
        -------
        list[ResultObject]
            The loaded results.
        """
        with open(self.path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def is_applicable(path):
        return path.suffix in (".pkl", ".pickle")


class NullStorageHandler(BaseStorageHandler):
    """Storage handler for no file access.

    Does not save or load any results, used when no path is provided.

    Saving has no effect, and ``load`` will always return empty data.

    Parameters
    ----------
    path : str, or Path coercible
        The path to the file to save to or load.
        Ignored, exists for API compatibility.
    """

    def save(self, results: list[ResultObject]):
        """Save the results - dummy method without any effect.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """
        return

    def _load(self) -> list[ResultObject]:
        """Load the results from a null file. Returns empty list.

        Returns
        -------
        list[ResultObject]
            The loaded results.
        """
        results = []
        return results

    @staticmethod
    def is_applicable(path):
        return path is None


STORAGE_HANDLERS = [
    NullStorageHandler,
    JSONStorageHandler,
    ParquetStorageHandler,
    CSVStorageHandler,
    PickleStorageHandler,
]


def get_storage_backend(path: str | Path) -> BaseStorageHandler:
    """Return the storage handler for a results file path.

    Selects a handler based on the file extension of ``path``. Pass
    ``None`` to obtain `NullStorageHandler` when no results file
    should be used.

    Parameters
    ----------
    path : str, pathlib.Path, or None
        Path to the results file. Must refer to a file, not a directory.
        Supported extensions are ``.json``, ``.csv``, and ``.parquet``.
        Use ``None`` when results should not be persisted to disk.

    Returns
    -------
    type[BaseStorageHandler]
        Handler class for ``path``. Instantiate it with ``path`` to
        read or write results.

    Raises
    ------
    ValueError
        If no handler supports the file extension of ``path``.
    """
    if isinstance(path, str):
        path = Path(path)
    for handler in STORAGE_HANDLERS:
        if handler.is_applicable(path):
            return handler
    raise ValueError(f"No storage handler found for {path}")


def _get_folds(row, extract_data=True):
    fold_infos = list(filter(lambda x: x.startswith("folds."), row.index))
    fold_ids = set(map(lambda x: x.split(".")[1], fold_infos))
    folds = {}
    for fold_id in fold_ids:
        fold_scores = row.filter(regex=f"folds.{fold_id}.scores.*")
        fold_gts = row.filter(regex=f"folds.{fold_id}.ground_truth")
        fold_predictions = row.filter(regex=f"folds.{fold_id}.predictions")
        fold_train_data = row.filter(regex=f"folds.{fold_id}.train_data")

        scores = {}

        unique_score = set(list(map(lambda x: x.split(".")[3], fold_scores.keys())))

        for score_name in unique_score:
            score_vals = row.filter(regex=f"folds.{fold_id}.scores.{score_name}.*")
            if len(score_vals) > 1:
                value = _create_df(score_vals)
            else:
                value = row[f"folds.{fold_id}.scores.{score_name}"]
            scores[score_name.split(".")[-1]] = value

        if len(fold_gts) > 0 and not fold_gts.isna().any():
            gt = _create_df(fold_gts)
        else:
            gt = None
        if len(fold_predictions) > 0 and not fold_predictions.isna().any():
            predictions = _create_df(fold_predictions)
        else:
            predictions = None

        if len(fold_train_data) > 0 and not fold_train_data.isna().any():
            train_data = _create_df(fold_train_data)
        else:
            train_data = None

        folds[int(fold_id)] = FoldResults(
            scores=scores,
            ground_truth=gt,
            predictions=predictions,
            train_data=train_data,
        )

    return folds


def _create_df(fold_gts):
    df_params = {}
    for name, value in fold_gts.items():
        if isinstance(value, str):
            df_params[name.split(".")[-1]] = ast.literal_eval(value)
        else:
            df_params[name.split(".")[-1]] = value
    df = pd.DataFrame.from_dict(df_params, orient="tight")
    return df.astype("float64")
