# ruff: noqa: S603

"""Run benchmark validation in an isolated environment subprocess."""

from __future__ import annotations

import builtins
import logging

import cloudpickle

from sktime.base import BaseEstimator
from sktime.benchmarking._benchmarking_dataclasses import TaskObject
from sktime.benchmarking._isolated_requirements import collect_pair_requirements
from sktime.utils.env_managers import BaseEnvironmentManager

logger = logging.getLogger(__name__)

_WORKER_MODULE = "sktime.benchmarking._worker"


def run_isolated_validation(
    *,
    benchmark_kind: str,
    task: TaskObject,
    estimator: BaseEstimator,
    backend,
    backend_params,
    return_data: bool,
    env_manager: BaseEnvironmentManager,
) -> dict:
    """Run ``_run_validation`` in a dedicated environment subprocess.

    Parameters
    ----------
    benchmark_kind : str
        Either ``"classification"`` or ``"forecasting"``.
    task : TaskObject
        Benchmark task definition.
    estimator : BaseEstimator
        Estimator to evaluate.
    backend : str or None
        Parallel backend passed to ``evaluate``.
    backend_params : dict or None
        Backend parameters passed to ``evaluate``.
    return_data : bool
        Whether fold predictions are returned.
    env_manager : BaseEnvironmentManager
        Manager used to create or reuse a pair environment and launch
        ``sktime.benchmarking._worker``.

    Returns
    -------
    dict
        Fold results mapping, as returned by ``_run_validation``.
    """
    requirements = collect_pair_requirements(estimator, task)
    payload = {
        "benchmark_kind": benchmark_kind,
        "task": task,
        "estimator": estimator,
        "backend": backend,
        "backend_params": backend_params,
        "return_data": return_data,
    }

    logger.debug(
        "Launching isolated validation (requirements: %s)",
        requirements or "core only",
    )

    proc = env_manager.run(
        _WORKER_MODULE,
        requirements=requirements,
        input=cloudpickle.dumps(payload),
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Isolated benchmark worker exited with code "
            f"{proc.returncode}: {stderr or 'no stderr output'}"
        )

    if not proc.stdout:
        raise RuntimeError("Isolated benchmark worker returned no output")

    result = cloudpickle.loads(proc.stdout)

    if result["status"] == "error":
        exc_type_name = result.get("exception_type", "Exception")
        exc_message = result.get("exception_message", "unknown error")
        traceback_text = result.get("traceback", "")
        if traceback_text:
            logger.debug("Isolated worker traceback:\n%s", traceback_text)

        exc_cls = getattr(builtins, exc_type_name, RuntimeError)
        if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
            raise exc_cls(exc_message)
        raise RuntimeError(f"{exc_type_name}: {exc_message}")

    return result["folds"]
