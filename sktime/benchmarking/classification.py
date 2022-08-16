# -*- coding: utf-8 -*-
"""Benchmarking for forecasting estimators."""
import functools
from typing import Callable, Dict, List, Optional, Union

from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.classification.base import BaseClassifier
from sktime.classification.model_evaluation._function import evaluate_classification
from sktime.forecasting.model_selection._split import BaseSplitter


def classification_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: List[str],
    classifier: BaseClassifier,
    **kwargs,
) -> Dict[str, Union[float, str]]:
    """Run validation for a forecasting estimator."""
    X, y = dataset_loader()

    results = {}
    for scorer in scorers:
        scores_df = evaluate_classification(
            classifier=classifier, X=X, y=y, cv=cv_splitter, scoring=scorer
        )
        for ix, row in scores_df.iterrows():
            results[f"{scorer}_fold_{ix}_test"] = row[f"{scorer}"]
        results[f"{scorer}_mean"] = scores_df[f"{scorer}"].mean()
        results[f"{scorer}_std"] = scores_df[f"{scorer}"].std()
    return results


def factory_classifying_validation(
    dataset_loader: Callable,
    cv_splitter: BaseSplitter,
    scorers: List[str],
) -> Callable:
    """Build validation func which just takes a forecasting estimator."""
    return functools.partial(
        classification_validation,
        dataset_loader,
        cv_splitter,
        scorers,
    )


class ClassificationBenchmark(BaseBenchmark):
    """Classification benchmark.

    Run a series of forecasters against a series of tasks defined via
    dataset loaders, cross validation splitting strategies and performance metrics,
    and return results as a df (as well as saving to file).
    """

    def add_task(
        self,
        dataset_loader: Callable,
        cv_splitter: BaseSplitter,
        scorers: List[str],
        task_id: Optional[str] = None,
    ):
        """Register a forecasting task to the benchmark."""
        task_kwargs = {
            "dataset_loader": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
        }
        if task_id is None:
            task_id = (
                f"[dataset={dataset_loader.__name__}]"
                f"_[cv_splitter={cv_splitter.__class__.__name__}]-v1"
            )
        self._add_task(factory_classifying_validation, task_kwargs, task_id=task_id)
