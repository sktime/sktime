"""Benchmarking for classification estimators."""

__author__ = ["jgyasu"]
__all__ = ["ClassificationBenchmark"]

from collections.abc import Callable
from typing import Any, Optional, Union

from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    TaskObject,
)
from sktime.benchmarking.benchmarks import (
    BaseBenchmark,
)
from sktime.classification.base import BaseClassifier
from sktime.classification.model_evaluation import evaluate


class ClassificationBenchmark(BaseBenchmark):
    """Classification benchmark."""

    def add_task(
        self,
        dataset_loader: Union[Callable, tuple],
        cv_splitter: Any,
        scorers: list,
        task_id: Optional[str] = None,
        error_score: str = "raise",
    ):
        """Register a classification task to the benchmark.

        Parameters
        ----------
        data : Union[Callable, tuple]
            Can be
            - a function which returns a dataset, like from `sktime.datasets`.
            - a tuple contianing two data container that are sktime comptaible.
            - single data container that is sktime compatible (only endogenous data).
        cv_splitter : BaseSplitter object
            Splitter used for generating validation folds.
        scorers : a list of BaseMetric objects
            Each BaseMetric output will be included in the results.
        task_id : str, optional (default=None)
            Identifier for the benchmark task. If none given then uses dataset loader
            name combined with cv_splitter class name.
        error_score : "raise" or numeric, default=np.nan
            Value to assign to the score if an exception occurs in estimator fitting.
            If set to "raise", the exception is raised. If a numeric value is given,
            FitFailedWarning is raised.

        Returns
        -------
        A dictionary of benchmark results for that classifier
        """
        if task_id is None:
            if hasattr(dataset_loader, "__name__"):
                task_id = (
                    f"[dataset={dataset_loader.__name__}]"
                    + f"_[cv_splitter={cv_splitter.__class__.__name__}]"
                )
            else:
                task_id = f"_[cv_splitter={cv_splitter.__class__.__name__}]"
        task_kwargs = {
            "data": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
            "error_score": error_score,
        }
        self._add_task(
            task_id,
            TaskObject(**task_kwargs),
        )

    def _run_validation(self, task: TaskObject, estimator: BaseClassifier):
        cv_splitter = task.cv_splitter
        scorers = task.scorers
        X, y = task.get_y_X()
        scores_df = evaluate(
            classifier=estimator,
            y=y,
            X=X,
            cv=cv_splitter,
            scoring=scorers,
            backend=self.backend,
            backend_params=self.backend_params,
            error_score=task.error_score,
            return_data=self.return_data,
        )

        folds = {}
        for ix, row in scores_df.iterrows():
            scores = {}
            for scorer in scorers:
                scores[scorer.__name__] = row["test_" + scorer.__name__]
            scores["fit_time"] = row["fit_time"]
            scores["pred_time"] = row["pred_time"]
            if self.return_data:
                folds[ix] = FoldResults(
                    scores, row["y_test"], row["y_pred"], row["y_train"]
                )
            else:
                folds[ix] = FoldResults(scores)
        return folds
