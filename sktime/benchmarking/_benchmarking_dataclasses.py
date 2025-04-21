"""Dataclasses for benchmarking."""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Optional, Union

import numpy as np
import pandas as pd

from sktime.benchmarking.base import BaseMetric
from sktime.split.base._base_splitter import BaseSplitter


def _coerce_data_for_evaluate(dataset_loader):
    """Coerce data input object to a dict to pass to forecasting evaluate."""
    if callable(dataset_loader) and not hasattr(dataset_loader, "load"):
        data = dataset_loader()
    elif callable(dataset_loader) and hasattr(dataset_loader, "load"):
        data = dataset_loader.load()
    else:
        data = dataset_loader

    if isinstance(data, tuple) and len(data) == 2:
        y, X = data
        return y, X
    elif isinstance(data, tuple) and len(data) == 1:
        return data[0], None
    else:
        return data, None


@dataclass
class TaskObject:
    """
    A forecasting task.

    Parameters
    ----------
    data: Union[Callable, tuple]
        Can be
        - a function which returns a dataset, like from `sktime.datasets`.
        - a tuple contianing two data container that are sktime comptaible.
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
    """

    data: Union[Callable, tuple]
    cv_splitter: BaseSplitter
    scorers: list[BaseMetric]
    strategy: str = "refit"
    cv_X = None
    cv_global: Optional[BaseSplitter] = None
    error_score: str = "raise"

    def get_y_X(self):
        """Get the endogenous and exogenous data."""
        return _coerce_data_for_evaluate(self.data)


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

    scores: list[dict[str, Union[float, pd.DataFrame]]]
    ground_truth: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    train_data: Optional[pd.DataFrame] = None

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
