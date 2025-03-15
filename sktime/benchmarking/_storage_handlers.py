"""Storage handlers for benchmarking results."""

import abc
import json
from pathlib import Path
from typing import Union

import pandas as pd

from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    ResultObject,
    asdict,
)


class BaseStorageHandler(abc.ABC):
    """Handles storage of benchmark results.

    The storage handler is responsible for storing and loading benchmark results.

    Parameters
    ----------
    path : str
        The path to the file to save to or load
    """

    def __init__(self, path):
        super().__init__()
        self.path = path

    @abc.abstractmethod
    def save(self, results: ResultObject):
        """Save the results to a file.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """

    @abc.abstractmethod
    def load(self) -> ResultObject:
        """Load the results from a file.

        Returns
        -------
        ResultObject
            The loaded results.

        """

    @staticmethod
    @abc.abstractmethod
    def is_applicable(path):
        """
        Check if the storage handler is applicable for the given path.

        Parameters
        ----------
        path : str
            The path to the file to save to or load

        Returns
        -------
        bool
            True if the storage handler is applicable for the given path.
        """


class JSONStorageHandler(BaseStorageHandler):
    """Storage handler for JSON files."""

    def save(self, results: list[ResultObject]):
        """Save the results to a JSON file.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """
        with open(self.path, "w") as f:
            json.dump(list(map(lambda x: asdict(x, pd_orient="list"), results)), f)

    def load(self) -> list[ResultObject]:
        """Load the results from a JSON file.

        Returns
        -------
        ResultObject
            The loaded results.

        """
        try:
            results = []
            with open(self.path) as f:
                results_json = json.load(f)
            for row in results_json:
                folds = {}
                for fold_id, fold in row["folds"].items():
                    scores = {}
                    for score_name, score_val in fold["scores"].items():
                        if isinstance(score_val, dict):
                            score_val = pd.DataFrame(score_val)
                        scores[score_name] = score_val
                    if "ground_truth" in fold:
                        ground_truth = pd.Series(fold["ground_truth"])
                    else:
                        ground_truth = None
                    if "predictions" in fold:
                        predictions = pd.Series(fold["predictions"])
                    else:
                        predictions = None
                    if "train_data" in fold:
                        train_data = pd.Series(fold["train_data"])
                    folds[int(fold_id)] = FoldResults(
                        scores, ground_truth, predictions, train_data
                    )

                results.append(
                    ResultObject(
                        model_id=row["model_id"],
                        validation_id=row["validation_id"],
                        folds=folds,
                    )
                )
            return results
        except FileNotFoundError:
            return []

    @staticmethod
    def is_applicable(path):
        return path.suffix == ".json"


class ParquetStorageHandler(BaseStorageHandler):
    """Storage handler for JSON files."""

    def save(self, results: list[ResultObject]):
        """Save the results to a JSON file.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """
        results_df = pd.json_normalize(
            list(map(lambda x: asdict(x, pd_orient="tight"), results))
        )

        results_df = results_df.sort_values(by=["validation_id", "model_id"])

        # TODO store fails in hierachical case with level report
        results_df = results_df.reset_index(drop=True)
        results_df.to_parquet(self.path, index=False)

    def load(self) -> list[ResultObject]:
        """Load the results from a JSON file.

        Returns
        -------
        ResultObject
            The loaded results.

        """
        try:
            results_df = pd.read_parquet(self.path)
            results = []
            for ix, row in results_df.iterrows():
                folds = {}
                for fold in row["folds"]:
                    scores = []
                    for score in fold["scores"]:
                        scores.append({score["name"]: score["score"]})
                    if "ground_truth" in fold:
                        ground_truth = pd.Series(fold["ground_truth"])
                    else:
                        ground_truth = None
                    if "predictions" in fold:
                        predictions = pd.Series(fold["predictions"])
                    else:
                        predictions = None
                    if "train_data" in fold:
                        train_data = pd.Series(fold["train_data"])
                    folds[fold["fold"]] = FoldResults(
                        scores, ground_truth, predictions, train_data
                    )

                results.append(
                    ResultObject(
                        model_id=row["model_id"],
                        validation_id=row["validation_id"],
                        folds=folds,
                    )
                )
            return results
        except FileNotFoundError:
            return []

    @staticmethod
    def is_applicable(path):
        return path.suffix == ".parquet"


class CSVStorageHandler(BaseStorageHandler):
    """Storage handler for JSON files."""

    def save(self, results: list[ResultObject]):
        """Save the results to a JSON file.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """
        results_df = pd.json_normalize(
            list(map(lambda x: asdict(x, pd_orient="tight"), results))
        )

        results_df = results_df.sort_values(by=["validation_id", "model_id"])

        # TODO store fails in hierachical case with level report
        results_df = results_df.reset_index(drop=True)
        results_df.to_csv(self.path, index=False)

    def _get_folds(self, row):
        fold_infos = list(filter(lambda x: x.startswith("folds."), row.index))
        fold_ids = set(map(lambda x: x.split(".")[1], fold_infos))
        folds = {}
        for fold_id in fold_ids:
            fold_scores = row.filter(regex=f"folds.{fold_id}.scores.*")

            scores = {}
            for score_name, value in fold_scores.items():
                scores[score_name.split(".")[-1]] = value

            folds[fold_id] = FoldResults(
                scores=scores,
            )
        return folds

    def load(self) -> list[ResultObject]:
        """Load the results from a JSON file.

        Returns
        -------
        ResultObject
            The loaded results.

        """
        try:
            results_df = pd.read_csv(self.path)
            results = []
            for ix, row in results_df.iterrows():
                folds = self._get_folds(row)
                results.append(
                    ResultObject(
                        model_id=row["model_id"],
                        validation_id=row["validation_id"],
                        folds=folds,
                    )
                )
            return results
        except FileNotFoundError:
            return []

    @staticmethod
    def is_applicable(path):
        return path.suffix == ".csv"


STORAGE_HANDLERS = [
    JSONStorageHandler,
    ParquetStorageHandler,
    CSVStorageHandler,
]


def get_storage_backend(path: Union[str, Path]) -> BaseStorageHandler:
    """Get the appropriate storage backend for a given path.

    Parameters
    ----------
    path : str
        The path to the file to save to or load

    Returns
    -------
    BaseStorageHandler
        The storage backend
    """
    for handler in STORAGE_HANDLERS:
        if handler.is_applicable(Path(path)):
            return handler
    raise ValueError(f"No storage handler found for {path}")
