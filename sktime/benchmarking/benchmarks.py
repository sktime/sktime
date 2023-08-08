"""Benchmarking interface for use with sktime objects.

Wraps kotsu benchmarking package.
"""
from typing import Callable, Optional, Union

import pandas as pd

from sktime.base import BaseEstimator
from sktime.utils.validation._dependencies import _check_soft_dependencies


def coer_estimator_and_id(estimators, estimator_id=None):
    """Coerce estimators to a dict with estimator_id as key and estimator as value.

    Parameters
    ----------
    estimators : dict, list or BaseEstimator object
        Estimator to coerce to a dict.
    estimator_id : str, optional (default=None)
        Identifier for estimator. If none given then uses estimator's class name.

    Returns
    -------
    estimators : dict
        Dict with estimator_id as key and estimator as value.
    """
    VERSION_ID = "-v1"
    if isinstance(estimators, dict):
        return estimators
    elif isinstance(estimators, list):
        return {
            f"{estimator.__class__.__name__ + VERSION_ID }": estimator
            for estimator in estimators
        }
    elif isinstance(estimators, BaseEstimator):
        estimator_id = estimator_id or f"{estimators.__class__.__name__+ VERSION_ID}"
        return {estimator_id: estimators}
    else:
        raise TypeError(
            "estimator must be of a type a dict, list or "
            f"BaseEstimator object but received {type(estimators)}"
        )


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
        estimator : Dict, List or BaseEstimator object
            Estimator to add to the benchmark.
            If Dict, keys are estimator_ids and values are estimators,
            use to customise the identifier ID.
            If List, each element is an estimator. estimator_ids are generated
            automatically using the estimator's class name.
        estimator_id : str, optional (default=None)
            Identifier for estimator. If none given then uses estimator's class name.
        """
        estimators = coer_estimator_and_id(estimator, estimator_id)
        for estimator_id, estimator in estimators.items():
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
