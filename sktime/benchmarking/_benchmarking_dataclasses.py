"""Dataclasses for benchmarking."""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from sktime.benchmarking.base import BaseMetric
from sktime.split.base._base_splitter import BaseSplitter
from sktime.split.singlewindow import SingleWindowSplitter


def _coerce_data_for_evaluate(dataset_loader, task_type=None):
    """Coerce data input object to a dict to pass to forecasting evaluate.

    Parameters
    ----------
    dataset_loader : Callable or tuple

        - a function which returns a dataset, like from `sktime.datasets`.
        - a tuple containing two data container that are sktime comptaible.
        - single data container that is sktime compatible (only first argument).

    task_type : str, optional (default=None)
        The type of task. One of "forecasting", "classification", "regression",
        "clustering". If None, X is assumed to be the first argument.

    Returns
    -------
    data_dict : dict
        A dictionary with keys "y" and "X" as appropriate for the task type,
        coerced from ``dataset_loader``.
    """
    if callable(dataset_loader) and not hasattr(dataset_loader, "load"):
        # Case 1: Loader function, e.g., load_longley
        data = dataset_loader()

    elif hasattr(dataset_loader, "load"):
        # Case 2: Dataset class or object, e.g., Longley or Longley()
        # if class, instantiate
        from inspect import isclass

        if isclass(dataset_loader):
            dataset_loader = dataset_loader()

        X = dataset_loader.load("X")
        y = dataset_loader.load("y")

        return {"X": X, "y": y}

    else:
        # Case 3: Data tuple or single data container
        data = dataset_loader

    if isinstance(data, tuple) and len(data) == 2:
        data0 = data[0]
        data1 = data[1]
    elif isinstance(data, tuple) and len(data) == 1:
        data0 = data[0]
        data1 = None
    else:
        data0 = data
        data1 = None

    if task_type == "forecasting":
        return {"y": data0, "X": data1}
    else:  # classification, regression, clustering
        return {"X": data0, "y": data1}


@dataclass
class TaskObject:
    """
    A benchmarking task.

    Parameters
    ----------
    data: Union[Callable, tuple]
        Can be

        - a function which returns a dataset, like from `sktime.datasets`.
        - a tuple containing two data container that are sktime comptaible.
        - single data container that is sktime compatible (only endogenous data).

    cv_splitter: BaseSplitter object
        Splitter used for generating validation folds.
    scorers: list of BaseMetric objects
        Each BaseMetric output will be included in the results.
    strategy: str, optional (default="refit")
        The strategy to use for refitting the model.
    cv_X: BaseSplitter object, optional (default=None)
        Splitter used for generating global cross-validation folds.
    cv_global: BaseSplitter object, optional (default=None)
        Splitter used for generating global cross-validation folds.
    error_score: str, optional (default="raise")
        The error score strategy to use.
    cv_global_temporal:  SingleWindowSplitter, default=None
        ignored if cv_global is None. If passed, it splits the Panel temporally
        before the instance split from cv_global is applied. This avoids
        temporal leakage in the global evaluation across time series.
        Has to be a SingleWindowSplitter.
        cv is applied on the test set of the combined application of
        cv_global and cv_global_temporal.
    """

    data: Callable | tuple
    cv_splitter: BaseSplitter
    scorers: list[BaseMetric]
    strategy: str = "refit"
    cv_X = None
    cv_global: BaseSplitter | None = None
    error_score: str = "raise"
    cv_global_temporal: SingleWindowSplitter | None = None

    def get_y_X(self, task_type=None):
        """Get the endogenous and exogenous data."""
        return _coerce_data_for_evaluate(self.data, task_type=task_type)


@dataclass
class FoldResults:
    """
    Results for a single fold.

    Parameters
    ----------
    scores: list of dicts with scorer name and score
        The scores for this fold for each scorer.
    ground_truth: pd.DataFrame, optional (default=None)
        The ground truth series for this fold.
    predictions: pd.DataFrame, optional (default=None)
        The predictions for this fold.
    train_data: pd.DataFrame, optional (default=None)
        The training data for this fold.
    """

    scores: list[dict[str, float | pd.DataFrame]]
    ground_truth: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    train_data: pd.DataFrame | None = None

    def __post_init__(self):
        """Check that scores are in the correct format."""
        for score_name, score_value in self.scores.items():
            if isinstance(score_value, pd.Series):
                self.scores[score_name] = score_value.to_frame()


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
    means : list of dict with scorer name and mean score
        The mean scores across all folds for each scorer.
    stds : list of dict with scorer name and standard deviation of the score
        The standard deviation of scores across all folds for
        each scorer.
    """

    model_id: str
    task_id: str
    folds: dict[int, FoldResults]
    means: dict[str, float] = field(init=False)
    stds: dict[str, float] = field(init=False)

    def __post_init__(self):
        """Calculate mean and std for each score."""
        self.means = {}
        self.stds = {}
        scores = {}
        for fold_idx, fold in self.folds.items():
            for score_name, score_value in fold.scores.items():
                if score_name not in scores:
                    scores[score_name] = []
                scores[score_name].append(score_value)
        for name, score in scores.items():
            if all(isinstance(s, (pd.DataFrame, pd.Series)) for s in score):
                score = pd.concat(score, axis=1)
                self.means[name] = np.mean(score, axis=1)
                self.stds[name] = np.std(score, ddof=1, axis=1)
            else:
                self.means[name] = np.mean(score, axis=0)
                self.stds[name] = np.std(score, axis=0, ddof=1)

    def to_dataframe(self):
        """Return results as a pandas DataFrame."""
        result_per_metric = {}
        gts = {}
        preds = {}
        train_data = {}
        for fold_idx, fold in self.folds.items():
            for score_name, score_value in fold.scores.items():
                if score_name not in result_per_metric:
                    result_per_metric[score_name] = {}
                result_per_metric[score_name][f"fold_{fold_idx}_test"] = score_value
            if fold.ground_truth is not None:
                gts[f"ground_truth_fold_{fold_idx}"] = [fold.ground_truth]
            if fold.predictions is not None:
                preds[f"predictions_fold_{fold_idx}"] = [fold.predictions]
            if fold.train_data is not None:
                train_data[f"train_data_fold_{fold_idx}"] = [fold.train_data]
        for metric, mean in self.means.items():
            result_per_metric[metric]["mean"] = mean
        for metric, std in self.stds.items():
            result_per_metric[metric]["std"] = std

        return pd.concat(
            [
                pd.DataFrame(
                    {"validation_id": [self.task_id], "model_id": [self.model_id]}
                ),
                pd.json_normalize(result_per_metric, sep="_"),
                pd.DataFrame(gts),
                pd.DataFrame(preds),
                pd.DataFrame(train_data),
            ],
            axis=1,
        )


def asdict(obj, *, dict_factory=dict, pd_orient="list"):
    """Return the fields of a dataclass as a dict.

    # Copied from dataclasses.asdict

    """
    if not hasattr(type(obj), "__dataclass_fields__"):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory, pd_orient)


def _asdict_inner(obj, dict_factory, pd_orient):
    # Copied from dataclasses._asdict_inner and slightly modified
    if hasattr(type(obj), "__dataclass_fields__"):
        result = []
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory, pd_orient)
            if f.name == "task_id":
                result.append(("validation_id", value))
            else:
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

        return type(obj)(*[_asdict_inner(v, dict_factory, pd_orient) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory, pd_orient) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (
                _asdict_inner(k, dict_factory, pd_orient),
                _asdict_inner(v, dict_factory, pd_orient),
            )
            for k, v in obj.items()
        )
    elif isinstance(obj, pd.Series):
        # With a frame, we have more control over the orientation.
        return obj.to_frame().to_dict(orient=pd_orient)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient=pd_orient)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return copy.deepcopy(obj)
