"""Benchmarking interface for use with sktime objects."""

from typing import Optional, Union

import pandas as pd

from sktime.base import BaseEstimator


def _is_initialised_estimator(estimator: BaseEstimator) -> bool:
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
    compatible = all(_is_initialised_estimator(estimator) for estimator in items)
    if not compatible:
        raise TypeError(
            "One or many estimator(s) is not an initialised BaseEstimator "
            "object(s). Please instantiate the estimator(s) first."
        )


def _coerce_estimator_and_id(estimators, estimator_id=None):
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
    elif _is_initialised_estimator(estimators):
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
        self.id_format = id_format

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
        estimators = _coerce_estimator_and_id(estimator, estimator_id)
        for estimator_id, estimator in estimators.items():
            self._add_estimator(estimator, estimator_id)

    def _add_estimator(
        self,
        estimator: BaseEstimator,
        estimator_id: Optional[str] = None,
    ):
        raise NotImplementedError(
            "Method not implemented in base class. "
            "Please implement this method in a subclass."
        )

    def _run(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented in base class.")

    def run(self, output_file: str, force_rerun: Union[str, list[str]] = "none"):
        """
        Run the benchmarking for all tasks and estimators.

        Parameters
        ----------
        output_file : str
            Path to save the results to.
        force_rerun : Union[str, list[str]], optional (default="none")
            If "none", will skip validation if results already exist.
            If "all", will run validation for all tasks and models.
            If list of str, will run validation for tasks and models in list.
        """
        return self._run(output_file, force_rerun)
