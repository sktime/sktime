"""Benchmarking for forecasting estimators."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.benchmarking._lib_mini_kotsu.run import _write
from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter


@dataclass
class TaskObject:
    """
    A forecasting task.

    Parameters
    ----------
    id: str
        The ID of the task.
    dataset_loader: Callable
        A function which returns a dataset, like from `sktime.datasets`.
    cv_splitter: BaseSplitter object
        Splitter used for generating validation folds.
    scorers: list of BaseMetric objects
        Each BaseMetric output will be included in the results.
    """

    id: str
    dataset_loader: Callable
    cv_splitter: BaseSplitter
    scorers: list[BaseMetric]


@dataclass
class ModelToTest:
    """
    A model to test.

    Parameters
    ----------
    id: str
        The ID of the model.
    model: BaseEstimator
        The model to test.
    """

    id: str
    model: BaseEstimator


@dataclass
class ScoreResult:
    """
    The result of a single scorer.

    Parameters
    ----------
    name: str
        The name of the scorer.
    score: float
        The score.
    """

    name: str
    score: float


@dataclass
class FoldResults:
    """
    Results for a single fold.

    Parameters
    ----------
    fold: int
        The fold number.
    scores: list of ScoreResult
        The scores for this fold for each scorer.
    ground_truth: pd.Series, optional (default=None)
        The ground truth series for this fold.
    predictions: pd.Series, optional (default=None)
        The predictions for this fold.
    """

    fold: int
    scores: list[ScoreResult]
    ground_truth: Optional[pd.Series] = None
    predictions: Optional[pd.Series] = None


@dataclass
class ResultObject:
    """
    Model results for a single task.

    Parameters
    ----------
    model_id : str
        The ID of the model.
    task_id : str
        The ID of the task.
    folds : list of FoldResults
        The results for each fold.
    means : list of ScoreResult
        The mean scores across all folds for each scorer.
    stds : list of ScoreResult
        The standard deviation of scores across all folds for
        each scorer.
    """

    model_id: str
    task_id: str
    folds: list[FoldResults]
    means: list[ScoreResult] = field(init=False)
    stds: list[ScoreResult] = field(init=False)

    def __post_init__(self):
        """Calculate mean and std for each score."""
        self.means = []
        self.stds = []
        scores = {}
        for fold in self.folds:
            for score in fold.scores:
                if score.name not in scores:
                    scores[score.name] = []
                scores[score.name].append(score.score)
        for name, score in scores.items():
            self.means.append(ScoreResult(name, np.mean(score)))
            self.stds.append(ScoreResult(name, np.std(score)))


class SktimeRegistry:
    """Register an entity by ID.

    IDs should remain stable over time and should be guaranteed to resolve to
    the same entity dynamics (or be desupported).
    """

    def __init__(self, entity_id_format: str = ""):
        self.entity_id_format = entity_id_format
        self.entities = {}

    def register(self, entity: Union[ModelToTest, TaskObject]):
        """Register an entity.

        Parameters
        ----------
        id: str
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
        if entity.id in self.entities.keys():
            # TODO implement that stuff
            warnings.warn(
                message=f"Entity with ID [id={id}] already registered, new id is: ...",
                category=UserWarning,
                stacklevel=2,
            )
        self.entities[entity.id] = entity


class ForecastingBenchmark(BaseBenchmark):
    """Forecasting benchmark.

    Run a series of forecasters against a series of tasks defined via dataset loaders,
    cross validation splitting strategies and performance metrics, and return results as
    a df (as well as saving to file).

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate for each task if specified.

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
        but changes the return to (lazy) ``dask.dataframe.DataFrame``.

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
    """

    def __init__(
        self,
        id_format: Optional[str] = None,
        backend=None,
        backend_params=None,
    ):
        super().__init__(id_format)
        self.backend = backend
        self.backend_params = backend_params
        self.estimators = SktimeRegistry()
        self.tasks = SktimeRegistry()

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
        estimator = estimator.clone()
        if estimator_id is None:
            estimator_id = estimator.__class__.__name__
        self.estimators.register(entity=ModelToTest(id=estimator_id, model=estimator))

    def _add_task(
        self,
        task: TaskObject,
    ):
        """Register a task to the benchmark."""
        self.tasks.register(
            entity=task,
        )

    def add_task(
        self,
        dataset_loader: Callable,
        cv_splitter: BaseSplitter,
        scorers: list[BaseMetric],
        task_id: Optional[str] = None,
    ):
        """Register a forecasting task to the benchmark.

        Parameters
        ----------
        dataset_loader : Callable
            A function which returns a dataset, like from `sktime.datasets`.
        cv_splitter : BaseSplitter object
            Splitter used for generating validation folds.
        scorers : a list of BaseMetric objects
            Each BaseMetric output will be included in the results.
        task_id : str, optional (default=None)
            Identifier for the benchmark task. If none given then uses dataset loader
            name combined with cv_splitter class name.

        Returns
        -------
        A dictionary of benchmark results for that forecaster
        """
        if task_id is None:
            task_id = (
                f"[dataset={dataset_loader.__name__}]"
                f"_[cv_splitter={cv_splitter.__class__.__name__}]"
            )
        task_kwargs = {
            "id": task_id,
            "dataset_loader": dataset_loader,
            "cv_splitter": cv_splitter,
            "scorers": scorers,
        }
        self._add_task(
            TaskObject(**task_kwargs),
        )

    def run(self, results_path: str, force_rerun: Union[str, list[str]] = "none"):
        """
        Run the benchmarking for all tasks and estimators.

        Parameters
        ----------
        results_path : str
            Path to save the results to.
        force_rerun : Union[str, list[str]], optional (default="none")
            If "all", rerun all tasks and estimators.
            If a list of estimator ids, rerun only those estimators.
            If "none", skip tasks and estimators that have already been run.
        """
        try:
            results_df = pd.read_csv(results_path)
        except FileNotFoundError:
            results_df = pd.DataFrame(
                columns=["validation_id", "model_id", "runtime_secs"]
            )
            results_df["runtime_secs"] = results_df["runtime_secs"].astype(int)

        results_df = results_df.set_index(["validation_id", "model_id"], drop=False)
        results_list = []

        for task in self.tasks.entities.values():
            for estimator in self.estimators.entities.values():
                if (
                    not force_rerun == "all"
                    and not (
                        isinstance(force_rerun, list) and estimator.id in force_rerun
                    )
                    and (task.id, estimator.id) in results_df.index
                ):
                    logging.info(
                        f"Skipping validation - model: "
                        f"{task.id} - {estimator.id}"
                        ", as found prior result in results."
                    )
                    continue

                logging.info(f"Running validation - model: {task.id} - {estimator.id}")

                results = self._run_validation(task, estimator)

                results_list.append(results)

        additional_results_df = pd.DataFrame.from_records(
            map(lambda x: asdict(x), results_list)
        )
        results_df = pd.concat([results_df, additional_results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(
            subset=["validation_id", "model_id"], keep="last"
        )
        results_df = results_df.sort_values(by=["validation_id", "model_id"])
        results_df = results_df.reset_index(drop=True)
        _write(
            results_df,
            results_path,
            to_front_cols=["validation_id", "model_id", "runtime_secs"],
        )
        return results_df

    def _run_validation(self, task: TaskObject, estimator: BaseForecaster):
        dataset_loader = task.dataset_loader
        cv_splitter = task.cv_splitter
        scorers = task.scorers
        y = dataset_loader()
        if isinstance(y, tuple):
            y, X = y
            scores_df = evaluate(
                forecaster=estimator.model,
                y=y,
                X=X,
                cv=cv_splitter,
                scoring=scorers,
                backend=self.backend,
                backend_params=self.backend_params,
                error_score="raise",
                return_data=True,
            )
        else:
            scores_df = evaluate(
                forecaster=estimator.model,
                y=y,
                cv=cv_splitter,
                scoring=scorers,
                backend=self.backend,
                backend_params=self.backend_params,
                error_score="raise",
                return_data=True,
            )

        folds = []
        for ix, row in scores_df.iterrows():
            scores = []
            for scorer in scorers:
                scores.append(ScoreResult(scorer.name, row["test_" + scorer.name]))
                scores.append(ScoreResult("fit_time", row["fit_time"]))
                scores.append(ScoreResult("pred_time", row["pred_time"]))
            folds.append(FoldResults(ix, scores, row["y_test"], row["y_pred"]))

        return ResultObject(estimator.id, task.id, folds)
