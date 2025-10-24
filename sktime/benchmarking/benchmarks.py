"""Benchmarking interface for use with sktime objects."""

import logging
import warnings
from dataclasses import dataclass, field

import pandas as pd

from sktime.base import BaseEstimator
from sktime.benchmarking._benchmarking_dataclasses import (
    ResultObject,
    TaskObject,
)
from sktime.benchmarking._storage_handlers import get_storage_backend
from sktime.benchmarking._utils import _check_id_format
from sktime.utils.unique_str import _make_strings_unique


def _is_initialised_estimator(estimator: BaseEstimator) -> bool:
    """Check if estimator is initialised BaseEstimator object."""
    if isinstance(estimator, BaseEstimator):
        return True
    return False


def _check_estimators_type(objs: dict | list | BaseEstimator) -> None:
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


@dataclass
class _BenchmarkingResults:
    """Results of a benchmarking run.

    Parameters
    ----------
    results : list of ResultObject
        The results of the benchmarking run.
    """

    path: str
    results: list[ResultObject] = field(default_factory=list)

    def __post_init__(self):
        """Load existing results from the path."""
        self.storage_backend = get_storage_backend(self.path)
        self.results = self.storage_backend(self.path).load()

    def update(self, new_result):
        """Update the results with a new result."""
        self.results.append(new_result)
        # todo: this should also update the storage backend!

    def save(self):
        """Save the results to a file."""
        self.storage_backend(self.path).save(self.results)

    def contains(self, task_id: str, model_id: str):
        """Check if the results contain a specific task and model.

        Parameters
        ----------
        task_id : str
            The task ID.
        model_id : str
            The model ID.
        """
        return any(
            result.task_id == task_id and result.model_id == model_id
            for result in self.results
        )

    def to_dataframe(self):
        """Convert the results to a pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        results_df = [result.to_dataframe() for result in self.results]
        df = pd.concat(results_df, axis=0, ignore_index=True)
        df["runtime_secs"] = df["pred_time_mean"] + df["fit_time_mean"]
        return df


class _SktimeRegistry:
    """Register an entity by ID.

    IDs should remain stable over time and should be guaranteed to resolve to
    the same entity dynamics (or be desupported).
    """

    def __init__(self, entity_id_format: str = ""):
        self.entity_id_format = entity_id_format
        self.entities = {}

    def register(self, entity_id, entity: BaseEstimator | TaskObject):
        """Register an entity.

        Parameters
        ----------
        entity_id: str
            A unique entity ID.
        entry_point: Callable or str
            The python entrypoint of the entity class. Should be one of:

            - the string path to the python object (e.g.module.name:factory_func, or
                module.name:Class)
            - the python object (class or factory) itself

        deprecated: Bool, optional (default=False)
            Flag to denote whether this entity should be skipped in validation runs
            and considered deprecated and replaced by a more recent/better model
        nondeterministic: Bool, optional (default=False)
            Whether this entity is non-deterministic even after seeding
        kwargs: Dict, optional (default=None)
            kwargs to pass to the entity entry point when instantiating the entity.
        """
        entity_id_unique = _make_strings_unique(list(self.entities.keys()), entity_id)
        _check_id_format(self.entity_id_format, entity_id_unique)
        if entity_id != entity_id_unique:
            warnings.warn(
                message=f"Entity with ID [id={entity_id}] already registered, "
                + "new id is {entity_id_unique}",
                category=UserWarning,
                stacklevel=2,
            )
        self.entities[entity_id_unique] = entity


class BaseBenchmark:
    """Base class for benchmarks.

    A benchmark consists of a set of tasks and a set of estimators.

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format
        if None, no format is enforced on task/estimator ID

    backend : string, by default "None".
        Parallelization backend to use for runs.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask", but changes the return to (lazy)
            ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the
        serialization backend (``cloudpickle``) for "dask" and "loky" is
        generally more robust than the standard ``pickle`` library used
        in "multiprocessing".

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by
          ``backend``. If ``n_jobs`` is not passed, it will default to ``-1``, other
          parameters will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

    return_data : bool, optional (default=False)
        Whether to return the prediction and the ground truth data in the results.
    """

    def __init__(
        self,
        id_format: str | None = None,
        backend=None,
        backend_params=None,
        return_data=False,
    ):
        self.id_format = id_format
        self.backend = backend
        self.backend_params = backend_params
        self.return_data = return_data
        self.estimators = _SktimeRegistry(id_format)
        self.tasks = _SktimeRegistry(id_format)

    def add_estimator(
        self,
        estimator: BaseEstimator,
        estimator_id: str | None = None,
    ):
        """Register an estimator to the benchmark.

        Parameters
        ----------
        estimator : dict, list or BaseEstimator object
            Estimator to add to the benchmark.

            * if ``BaseEstimator``, single estimator. ``estimator_id`` is generated
              as the estimator's class name if not provided.
            * If ``dict``, keys are ``estimator_id``s used to customise identifier ID
              and values are estimators.
            * If ``list``, each element is an estimator. ``estimator_id``s are generated
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
        estimator_id: str | None = None,
    ):
        """Register a single estimator to the benchmark.

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
        estimator = estimator.clone()
        if estimator_id is None:
            estimator_id = estimator.__class__.__name__
        self.estimators.register(entity_id=estimator_id, entity=estimator)

    def _add_task(self, task_id: str, task: TaskObject):
        """Register a task to the benchmark."""
        self.tasks.register(entity_id=task_id, entity=task)

    def add_task(self, *args, **kwargs):
        """Register a task to the benchmark."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    def _run(self, results_path: str, force_rerun: str | list[str] = "none"):
        """
        Run the benchmarking for all tasks and estimators.

        Parameters
        ----------
        results_path : str
            Path to save the results to.
            If None, will not save the results.

        force_rerun : Union[str, list[str]], optional (default="none")

            * If "none", will skip validation if results already exist.
            * If "all", will run validation for all tasks and models.
            * If list of str, will run validation for tasks and models in list.
        """
        results = _BenchmarkingResults(path=results_path)

        for task_id, estimator_id, task, estimator in self._generate_experiments():
            if results.contains(task_id, estimator_id) and (
                force_rerun == "none"
                or (isinstance(force_rerun, list) and estimator_id not in force_rerun)
            ):
                logging.info(
                    f"Skipping validation - model: "
                    f"{task_id} - {estimator_id}"
                    ", as found prior result in results."
                )
                continue

            logging.info(f"Running validation - model: {task_id} - {estimator_id}")
            folds = self._run_validation(task, estimator)
            results.update(
                ResultObject(
                    task_id=task_id,
                    model_id=estimator_id,
                    folds=folds,
                )
            )

        if results_path is not None:
            results.save()
        return results.to_dataframe()

    def _generate_experiments(self):
        """Generate experiments for the benchmark.

        Returns a list of tuples with:

        * task_id: str
        * estimator_id: str
        * task: TaskObject
        * estimator: estimator object
        """
        tasks = self.tasks.entities
        estimators = self.estimators.entities
        exps = []
        for task_id, task in tasks.items():
            for estimator_id, estimator in estimators.items():
                exps.append((task_id, estimator_id, task, estimator))
        return exps

    def run(self, output_file: str, force_rerun: str | list[str] = "none"):
        """
        Run the benchmarking for all tasks and estimators.

        Parameters
        ----------
        output_file : str or None.
            Path to save the results to.
            If None, results will not be saved.

        force_rerun : Union[str, list[str]], optional (default="none")

            * If "none", will skip validation if results already exist.
            * If "all", will run validation for all tasks and models.
            * If list of str, will run validation for tasks and models in list.
        """
        return self._run(output_file, force_rerun)

    def _run_validation(self, task: TaskObject, estimator: BaseEstimator):
        """Run validation for a single task and estimator."""
        raise NotImplementedError("This method must be implemented by a subclass.")
