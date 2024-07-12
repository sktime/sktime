"""Benchmarking interface for use with sktime objects.

Wraps kotsu benchmarking package.
"""

from typing import Callable, Optional, Union

import pandas as pd

from sktime.base import BaseEstimator


# TODO: typo but need to be deprecated
# See https://www.sktime.net/en/stable/developer_guide/deprecation.html
def is_initalised_estimator(estimator: BaseEstimator) -> bool:
    """Check if estimator is initialised BaseEstimator object."""
    if isinstance(estimator, BaseEstimator):
        return True
    return False


def _check_estimators_type(objs: Union[dict, list, BaseEstimator]) -> None:
    """Check if all estimators are initialised BaseEstimator objects.

    Raises
    ------
    TypeError
        If any of the estimators are not BaseEstimator objects.
    """
    if isinstance(objs, BaseEstimator):
        objs = [objs]
    items = objs.values() if isinstance(objs, dict) else objs
    compatible = all(is_initalised_estimator(estimator) for estimator in items)
    if not compatible:
        raise TypeError(
            "One or many estimator(s) is not an initialised BaseEstimator "
            "object(s). Please instantiate the estimator(s) first."
        )


def coerce_estimator_and_id(estimators, estimator_id=None):
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
    _check_estimators_type(estimators)
    if isinstance(estimators, dict):
        return estimators
    elif isinstance(estimators, list):
        return {estimator.__class__.__name__: estimator for estimator in estimators}
    elif is_initalised_estimator(estimators):
        estimator_id = estimator_id or estimators.__class__.__name__
        return {estimator_id: estimators}
    else:
        raise TypeError(
            "estimator must be of a type a dict, list or an initialised "
            f"BaseEstimator object but received {type(estimators)} type."
        )


class BaseBenchmark:
    """Base class for benchmarks.

    A benchmark consists of a set of tasks and a set of estimators.

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format
        if None, no format is enforced on task/estimator ID

    """

    def __init__(self, id_format: Optional[str] = None):
        from sktime.benchmarking._base_kotsu import (
            SktimeModelRegistry,
            SktimeValidationRegistry,
        )
        from sktime.benchmarking._lib_mini_kotsu.run import run

        self.estimators = SktimeModelRegistry(id_format)
        self.validations = SktimeValidationRegistry(id_format)
        self.kotsu_run = run

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
            If Dict, keys are estimator_ids used to customise identifier ID
            and values are estimators.
            If List, each element is an estimator. estimator_ids are generated
            automatically using the estimator's class name.
        estimator_id : str, optional (default=None)
            Identifier for estimator. If none given then uses estimator's class name.
        """
        estimators = coerce_estimator_and_id(estimator, estimator_id)
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
            f"{task_entrypoint}"
            if isinstance(task_entrypoint, str)
            else f"{task_entrypoint.__name__}"
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
