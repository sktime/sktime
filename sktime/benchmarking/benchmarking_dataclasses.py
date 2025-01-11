"""Dataclasses for benchmarking."""

import abc
import copy
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import pandas as pd

from build.lib.sktime.split.base._base_splitter import BaseSplitter
from sktime.base._base import BaseEstimator
from sktime.benchmarking.base import BaseMetric


@dataclass
class TaskObject:
    """
    A forecasting task.

    Parameters
    ----------
    id: str
        The ID of the task.
    dataset_loader: Callable
        A function which returns a dataset, like from `sktime.datasets`.
    cv_splitter: BaseSplitter object
        Splitter used for generating validation folds.
    scorers: list of BaseMetric objects
        Each BaseMetric output will be included in the results.
    """

    id: str
    dataset_loader: Callable
    cv_splitter: BaseSplitter
    scorers: list[BaseMetric]


@dataclass
class ModelToTest:
    """
    A model to test.

    Parameters
    ----------
    id: str
        The ID of the model.
    model: BaseEstimator
        The model to test.
    """

    id: str
    model: BaseEstimator


@dataclass
class ScoreResult:
    """
    The result of a single scorer.

    Parameters
    ----------
    name: str
        The name of the scorer.
    score: float
        The score.
    """

    name: str
    score: float


@dataclass
class FoldResults:
    """
    Results for a single fold.

    Parameters
    ----------
    fold: int
        The fold number.
    scores: list of ScoreResult
        The scores for this fold for each scorer.
    ground_truth: pd.Series, optional (default=None)
        The ground truth series for this fold.
    predictions: pd.Series, optional (default=None)
        The predictions for this fold.
    """

    fold: int
    scores: list[ScoreResult]
    ground_truth: Optional[pd.Series] = None
    predictions: Optional[pd.Series] = None
    train_data: Optional[pd.Series] = None


@dataclass
class ResultObject:
    """
    Model results for a single task.

    Parameters
    ----------
    model_id : str
        The ID of the model.
    task_id : str
        The ID of the task.
    folds : list of FoldResults
        The results for each fold.
    means : list of ScoreResult
        The mean scores across all folds for each scorer.
    stds : list of ScoreResult
        The standard deviation of scores across all folds for
        each scorer.
    """

    model_id: str
    task_id: str
    folds: list[FoldResults]
    means: list[ScoreResult] = field(init=False)
    stds: list[ScoreResult] = field(init=False)

    def __post_init__(self):
        """Calculate mean and std for each score."""
        self.means = []
        self.stds = []
        scores = {}
        for fold in self.folds:
            for score in fold.scores:
                if score.name not in scores:
                    scores[score.name] = []
                scores[score.name].append(score.score)
        for name, score in scores.items():
            self.means.append(ScoreResult(name, np.mean(score)))
            self.stds.append(ScoreResult(name, np.std(score)))


@dataclass
class BenchmarkingResults:
    """Results of a benchmarking run.

    Parameters
    ----------
    results : list of ResultObject
        The results of the benchmarking run.
    """

    path: str
    results: list[ResultObject] = field(default_factory=list)

    def __post_init__(self):
        """Save the results to a file."""
        self.storage_backend = get_storage_backend(self.path)
        self.results = self.storage_backend(self.path).load()

    def save(self):
        """Save the results to a file."""
        self.storage_backend(self.path).save(self.results)

    def contains(self, task_id: str, model_id: str):
        """
        Check if the results contain a specific task and model.

        Parameters
        ----------
        task_id : str
            The task ID.
        model_id : str
            The model ID.
        """
        return any(
            [
                result.task_id == task_id and result.model_id == model_id
                for result in self.results
            ]
        )


def asdict(obj, *, dict_factory=dict):
    """Return the fields of a dataclass as a dict.

    # Copied from dataclasses.asdict

    """
    if not hasattr(type(obj), "__dataclass_fields__"):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_inner(obj, dict_factory):
    # Copied from dataclasses._asdict_inner and slightly modified
    if hasattr(type(obj), "__dataclass_fields__"):
        result = []
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
            for k, v in obj.items()
        )
    elif isinstance(obj, pd.Series):
        return obj.to_json()
    else:
        return copy.deepcopy(obj)


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


class ParquetStorageHandler(BaseStorageHandler):
    """Storage handler for JSON files."""

    def save(self, results: list[ResultObject]):
        """Save the results to a JSON file.

        Parameters
        ----------
        results : ResultObject
            The results to save.
        """
        results_df = pd.DataFrame.from_records(map(lambda x: asdict(x), results))
        # results_df = pd.concat([results_df, additional_results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(
            subset=["task_id", "model_id"], keep="last"
        )
        results_df = results_df.sort_values(by=["task_id", "model_id"])
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
                folds = []
                for fold in row["folds"]:
                    scores = []
                    for score in fold["scores"]:
                        scores.append(ScoreResult(score["name"], score["score"]))
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
                    folds.append(
                        FoldResults(
                            fold["fold"], scores, ground_truth, predictions, train_data
                        )
                    )

                results.append(
                    ResultObject(
                        model_id=row["model_id"],
                        task_id=row["task_id"],
                        folds=folds,
                    )
                )
            return results
        except FileNotFoundError:
            return []


def get_storage_backend(path: str) -> BaseStorageHandler:
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
    if path.endswith(".parquet"):
        return ParquetStorageHandler
    else:
        raise ValueError(f"Unsupported file format: {path}")
