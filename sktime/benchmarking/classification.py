"""Benchmarking for classification estimators."""

__author__ = ["jgyasu"]
__all__ = ["ClassificationBenchmark"]

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd

from sktime.base import BaseEstimator
from sktime.benchmarking._benchmarking_dataclasses import (
    FoldResults,
    ResultObject,
    TaskObject,
)
from sktime.benchmarking._storage_handlers import get_storage_backend
from sktime.benchmarking._utils import _check_id_format
from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.classification.base import BaseClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.utils.unique_str import _make_strings_unique


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
        """Save the results to a file."""
        self.storage_backend = get_storage_backend(self.path)
        self.results = self.storage_backend(self.path).load()

    def save(self):
        """Save the results to a file."""
        self.storage_backend(self.path).save(self.results)

    def contains(self, task_id: str, model_id: str):
        """
        Check if the results contain a specific task and model.

        Parameters
        ----------
        task_id : str
            The task ID.
        model_id : str
            The model ID.
        """
        return any(
            [
                result.task_id == task_id and result.model_id == model_id
                for result in self.results
            ]
        )

    def to_dataframe(self):
        """Convert the results to a pandas DataFrame."""
        results = []
        for result in self.results:
            results.append(result.to_dataframe())

        df = pd.concat(results, axis=0, ignore_index=True)
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

    def register(self, entity_id, entity: Union[BaseEstimator, TaskObject]):
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


class ClassificationBenchmark(BaseBenchmark):
    """Classification benchmark.

    Run a series of classifiers against a series of tasks defined via dataset loaders,
    cross validation splitting strategies and performance metrics, and return results as
    a df (as well as saving to file).

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format

    backend : string, by default "None".
        Parallelization backend to use for runs.

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
        but changes the return to (lazy) ``dask.dataframe.DataFrame``.
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
        id_format: Optional[str] = None,
        backend=None,
        backend_params=None,
        return_data=False,
    ):
        super().__init__(id_format)
        self.backend = backend
        self.backend_params = backend_params
        self.estimators = _SktimeRegistry(id_format)
        self.tasks = _SktimeRegistry(id_format)
        self.return_data = return_data

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
        if isinstance(estimator, dict):
            for key, value in estimator.items():
                self.add_estimator(value, key)
        elif isinstance(estimator, list):
            for value in estimator:
                self.add_estimator(value)
        else:
            self._add_estimator(estimator, estimator_id)

    def _add_estimator(
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
        estimator = estimator.clone()
        if estimator_id is None:
            estimator_id = estimator.__class__.__name__
        self.estimators.register(entity_id=estimator_id, entity=estimator)

    def _add_task(
        self,
        task_id: str,
        task: TaskObject,
    ):
        """Register a task to the benchmark."""
        self.tasks.register(
            entity_id=task_id,
            entity=task,
        )

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

    def _run(self, results_path: str, force_rerun: Union[str, list[str]] = "none"):
        """
        Run the benchmarking for all tasks and estimators.

        Parameters
        ----------
        results_path : str
            Path to save the results to.
        force_rerun : Union[str, list[str]], optional (default="none")
            If "none", will skip validation if results already exist.
            If "all", will run validation for all tasks and models.
            If list of str, will run validation for tasks and models in list.
        """
        results = _BenchmarkingResults(path=results_path)

        for task_id, task in self.tasks.entities.items():
            for estimator_id, estimator in self.estimators.entities.items():
                if results.contains(task_id, estimator_id) and (
                    force_rerun == "none"
                    or (
                        isinstance(force_rerun, list)
                        and estimator_id not in force_rerun
                    )
                ):
                    logging.info(
                        f"Skipping validation - model: "
                        f"{task_id} - {estimator_id}"
                        ", as found prior result in results."
                    )
                    continue

                logging.info(f"Running validation - model: {task_id} - {estimator_id}")
                folds = self._run_validation(task, estimator)
                results.results.append(
                    ResultObject(
                        task_id=task_id,
                        model_id=estimator_id,
                        folds=folds,
                    )
                )

        results.save()
        return results.to_dataframe()

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
