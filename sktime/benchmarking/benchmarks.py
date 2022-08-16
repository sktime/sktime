# -*- coding: utf-8 -*-
"""Benchmarking interface for use with sktime objects.

Wraps kotsu benchmarking package.
"""
from typing import Callable, Optional, Union

import pandas as pd

from sktime.base import BaseEstimator
from sktime.utils.validation._dependencies import _check_soft_dependencies


class BaseBenchmark:
    """Base class for benchmarks.

    A benchmark consists of a set of tasks and a set of estimators.
    """

    def __init__(self):
        _check_soft_dependencies("kotsu")
        import kotsu

        self.estimators = kotsu.registration.ModelRegistry()
        self.validations = kotsu.registration.ValidationRegistry()
        self.kotsu_run = kotsu.run.run

    def add_estimator(
        self,
        estimator: BaseEstimator,
        estimator_id: Optional[str] = None,
    ):
        """Register an estimator to the benchmark.

        Parameters
        ----------
        estimator : BaseEstimator object
            Estimator to add to the benchmark.
        estimator_id : str, optional (default=None)
            Identifier for estimator. If none given then uses estimator's class name.

        """
        estimator_id = estimator_id or f"{estimator.__class__.__name__}-v1"
        estimator = estimator.clone()  # extra cautious
        self.estimators.register(id=estimator_id, entry_point=estimator.clone)

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

    def run(self, output_file: str) -> pd.DataFrame:
        """Run the benchmark.

        Parameters
        ----------
        output_file : str
            Path to write results output file to.

        Returns
        -------
        pandas DataFrame of results
        """
        results_df = self.kotsu_run(self.estimators, self.validations, output_file)
        return results_df
