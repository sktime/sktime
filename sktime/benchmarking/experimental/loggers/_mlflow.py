"""Uniform interface of different experiment tracking packages."""

import os
from typing import Any, Dict, Optional

from sktime.base import BaseEstimator
from sktime.benchmarking.experimental.loggers._base import BaseLogger
from sktime.utils.validation._dependencies import _check_soft_dependencies

# mlflavour has no seperation of dependencies.
if _check_soft_dependencies("mlflow", severity="warning"):
    import mlflow
    from mlflow.client import MlflowClient

    from sktime.utils import mlflow_sktime

LOCAL_URI_PREFIX = "file:"
DEFAULT_ENV = os.getenv("MLFLOW_TRACKING_URI")


class MLFlowLogger(BaseLogger):
    """Manage the interface for tracking using Mlflow."""

    def __init__(
        self,
        experiment_name: str = "benchmark_logs",
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = "./benchmarkruns",
        tracking_uri: Optional[str] = DEFAULT_ENV,
        artifact_location: Optional[str] = None,
    ):
        _check_soft_dependencies("mlflow", severity="error")
        if tracking_uri is None:
            self._tracking_uri = f"{LOCAL_URI_PREFIX}{save_dir}"

        self._experiment_name = experiment_name
        self.tags = tags
        self._save_dir = save_dir
        self._artifact_location = artifact_location

        self._has_experiment = False
        self._client = MlflowClient(tracking_uri)

    @property
    def save_dir(self) -> str:
        """Root directory for saving MLflow experiments.

        Returns
        -------
        Local path to the root experiment directory if the tracking URI is local.
        Otherwise returns `None`. Check
        """
        if self._tracking_uri.startswith(LOCAL_URI_PREFIX):
            return self._tracking_uri.lstrip(LOCAL_URI_PREFIX)
        return None

    def _start_experiment(self) -> None:
        self._has_experiment = True
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client.create_experiment(
            name=self._experiment_name,
            artifact_location=self._artifact_location,
            tags=self.tags,
        )

    def start_run(
        self, run_name: Optional[str] = None, run_id: Optional[str] = None
    ) -> None:
        """Start a new active run. If there is an active run, shut it down first."""
        if not self._has_initialised:
            self._start_experiment()

        if mlflow.active_run() is not None:
            self.end_run()

        mlflow.start_run(run_name=run_name, run_id=run_id)

    def log_metric(self, metric_name: str, metric_values: float) -> None:
        """Log a metric under the current run."""
        mlflow.log_metric(metric_name, metric_values)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics for the current run."""
        mlflow.log_metrics(metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log a batch of estimator params for the current run."""
        mlflow.log_params(params)

    def log_estimator(self, estimator: BaseEstimator) -> None:
        """Save estimator state as an artifact."""
        mlflow_sktime.log_model(
            sktime_model=estimator,
            artifact_path=self._artifact_location,
            registered_model_name=estimator.__class__.__name__,
        )

    def log_graph(self, graph, save_kwargs: Dict["str", Any]) -> None:
        """Log a figure as an artifact.

        The following figure objects are supported:
        - matplotlib.figure.Figure_
        - plotly.graph_objects.Figure_
        """
        mlflow.log_figure(graph, self._artifact_location, save_kwargs=save_kwargs)

    def log_cvsplit(self, cvsplit_info: Dict[str, Any]) -> None:
        """Log cvsplit metadata as an artifact."""
        mlflow.log_dict(cvsplit_info, self._artifact_location)

    def get_estimator(self, estimator_id) -> None:
        """Fetch estimator from artifact."""
        path = f"{self._artifact_location}/{estimator_id}"
        mlflow_sktime.load_model(path)

    def end_run(self) -> None:
        """End an active run."""
        mlflow.end_run()
