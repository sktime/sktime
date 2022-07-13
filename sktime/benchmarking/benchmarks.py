# -*- coding: utf-8 -*-
"""Benchmarking interface for use with sktime objects.

Wraps kotsu benchmarking interface.
"""
from typing import Callable, Optional, Type, Union

import kotsu

from sktime.base import BaseEstimator


class BaseBenchmark:
    """Base class for benchmarks.

    A benchmark consists of a set of tasks and a set of estimators.
    """

    def __init__(self):
        self.estimators = kotsu.registration.ModelRegistry()
        self.validations = kotsu.registration.ValidationRegistry()

    def add_estimator(
        self,
        estimator_entrypoint: Type[BaseEstimator],
        estimator_kwargs: Optional[dict] = None,
        estimator_id: Optional[str] = None,
    ):
        """Register an estimator to the benchmark."""
        estimator_id = estimator_id or f"{estimator_entrypoint.__name__}-v1"
        self.estimators.register(
            id=estimator_id, entry_point=estimator_entrypoint, kwargs=estimator_kwargs
        )

    def _add_task(
        self,
        task_entrypoint: Union[Callable, str],
        task_kwargs: Optional[dict] = None,
        task_id: Optional[str] = None,
    ):
        """Register a task to the benchmark."""
        task_id = task_id or (
            f"{task_entrypoint}-v1"
            if isinstance(task_entrypoint, str)
            else f"{task_entrypoint.__name__}-v1"
        )
        self.validations.register(
            id=task_id, entry_point=task_entrypoint, kwargs=task_kwargs
        )

    def run(self, output_file):
        """Run the benchmark."""
        results_df = kotsu.run.run(self.estimators, self.validations, output_file)
        return results_df
