"""Benchmarking interface for use with sktime objects.

Wraps kotsu benchmarking package.
"""

import time
from collections.abc import Callable
from typing import Optional, Union

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

        self.estimators = SktimeModelRegistry(id_format)
        self.validations = SktimeValidationRegistry(id_format)

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

    def run(self, output_file=None) -> pd.DataFrame:
        """Run the benchmark.

        Parameters
        ----------
        output_file : str or None
            If not provided, will not write results to file.
            If provided, will write the output in csv format to the given path.
            Paths are relative to the current working directory.

        Returns
        -------
        pandas DataFrame of results
        """
        results_df = self._run(output_file)
        return results_df

    def _run(self, results_path: str = "./validation_results.csv"):
        """Run a registry of models through a registry of validations.

        Parameters
        ----------
        results_path: string, default = "./validation_results.csv"
            The file path to which the results will be written to, and results from prior
            runs will be read from.

        Returns
        -------
        pd.DataFrame: dataframe of validation results.
        """
        model_registry = self.estimators
        validation_registry = self.validations

        results_df = _load(results_path)  # returns None if does not exist

        if results_df is None:
            results_df = pd.DataFrame(
                columns=["validation_id", "model_id", "runtime_secs"]
            )
            results_df["runtime_secs"] = results_df["runtime_secs"].astype(int)

        results_df = results_df.set_index(["validation_id", "model_id"], drop=False)
        results_list = []

        for validation_spec in validation_registry.all():
            for model_spec in model_registry.all():
                validation = validation_spec.make()

                model = model_spec.make()
                results, elapsed_secs = self._run_validation_model(validation, model)
                results = self._add_meta_data_to_results(
                    results, elapsed_secs, validation_spec, model_spec
                )
                results_list.append(results)

        additional_results_df = pd.DataFrame.from_records(results_list)
        results_df = pd.concat([results_df, additional_results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(
            subset=["validation_id", "model_id"], keep="last"
        )
        results_df = results_df.sort_values(by=["validation_id", "model_id"])
        results_df = results_df.reset_index(drop=True)
        if results_path is not None:
            _write(
                results_df,
                results_path,
                to_front_cols=["validation_id", "model_id", "runtime_secs"],
            )
        return results_df

    def _run_validation_model(self, validation, model):
        """Run given validation on given model, and store the results.

        Returns
        -------
        A tuple of (dict of results: Results type, elapsed time in seconds)
        """
        start_time = time.time()
        results = validation(model)
        elapsed_secs = time.time() - start_time
        return results, elapsed_secs

    def _add_meta_data_to_results(
        self,
        results,
        elapsed_secs: float,
        validation_spec,
        model_spec,
    ):
        """Add meta data to results, raising if keys clash.

        Parameters
        ----------
        results: dict
            The results to add the meta data to.
        elapsed_secs: float
            The elapsed time in seconds.
        validation_spec: ValidationSpec
            The validation spec, id is added to the results.
        model_spec: ModelSpec
            The model spec, id is addes to the results.
        """
        results_meta_data = {
            "validation_id": validation_spec.id,
            "model_id": model_spec.id,
            "runtime_secs": elapsed_secs,
        }
        if bool(set(results) & set(results_meta_data)):
            raise ValueError(
                f"Validation:{validation_spec.id} on model:{model_spec.id} "
                f"returned results:{results} which contains a privileged key name."
            )
        return {**results, **results_meta_data}


def _write(df: pd.DataFrame, results_path: str, to_front_cols: list[str]):
    """Write the results to the results path."""
    df = df[to_front_cols + [col for col in df.columns if col not in to_front_cols]]
    df.to_csv(results_path, index=False)


def _load(results_path: str):
    """Load results from file if it exists."""
    if results_path is None:
        return None
    try:
        return pd.read_csv(results_path)
    except FileNotFoundError:
        return None
