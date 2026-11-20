"""Run benchmark validation in an isolated environment subprocess."""

from __future__ import annotations

import builtins
import logging
import traceback

import cloudpickle

from sktime.base import BaseEstimator
from sktime.benchmarking._benchmarking_dataclasses import TaskObject
from sktime.benchmarking._isolated_requirements import collect_pair_requirements
from sktime.utils._env_managers import BaseEnvironmentManager

logger = logging.getLogger(__name__)


def _execute_validation(benchmark_cls, init_params, task, estimator):
    """Build a fresh benchmark and call its ``_run_validation``.

    Module-level so the child process receives this function, not a bound
    method of the parent benchmark. A bound method would pickle the parent
    and every task and estimator registered on it.

    Parameters
    ----------
    benchmark_cls : type
        Benchmark class to instantiate. Any subclass of ``BaseBenchmark``
        whose ``_run_validation`` is implemented.
    init_params : dict
        Constructor arguments for ``benchmark_cls``. Must not turn isolation
        back on.
    task : TaskObject
        Single task to validate.
    estimator : estimator
        Single estimator to validate.

    Returns
    -------
    dict
        ``{"status": "success", "folds": ...}`` or
        ``{"status": "error", ...}``.
    """
    benchmark = benchmark_cls(**init_params)
    try:
        folds = benchmark._run_validation(task, estimator)
    except Exception as exc:
        return {
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return {"status": "success", "folds": folds}


def run_isolated_validation(
    *,
    benchmark_cls: type,
    init_params: dict,
    task: TaskObject,
    estimator: BaseEstimator,
    env_manager: BaseEnvironmentManager,
) -> dict:
    """Run ``benchmark_cls._run_validation`` in an isolated environment.

    Parameters
    ----------
    benchmark_cls : type
        Benchmark class whose ``_run_validation`` performs the validation.
        The child instantiates this class; it does not receive the parent
        instance.
    init_params : dict
        Constructor arguments for a fresh ``benchmark_cls`` in the child.
    task : TaskObject
        Benchmark task definition.
    estimator : BaseEstimator
        Estimator to evaluate.
    env_manager : BaseEnvironmentManager
        Manager used to create or reuse a pair environment and run
        ``_execute_validation``.

    Returns
    -------
    dict
        Fold results mapping, as returned by ``_run_validation``.
    """
    requirements = collect_pair_requirements(estimator, task)

    logger.debug(
        "Launching isolated validation (requirements: %s)",
        requirements or "core only",
    )

    proc = env_manager.run(
        _execute_validation,
        requirements=requirements,
        args=(benchmark_cls, init_params, task, estimator),
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
