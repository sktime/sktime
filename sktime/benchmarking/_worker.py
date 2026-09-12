"""Subprocess worker for isolated benchmark validation."""

from __future__ import annotations

import sys
import traceback


def _run_validation(payload: dict):
    """Execute validation for a single task-estimator pair."""
    benchmark_kind = payload["benchmark_kind"]
    task = payload["task"]
    estimator = payload["estimator"]
    backend = payload.get("backend")
    backend_params = payload.get("backend_params")
    return_data = payload.get("return_data", False)

    if benchmark_kind == "classification":
        from sktime.benchmarking.classification import ClassificationBenchmark

        benchmark = ClassificationBenchmark(
            backend=backend,
            backend_params=backend_params,
            return_data=return_data,
        )
    elif benchmark_kind == "forecasting":
        from sktime.benchmarking.forecasting import ForecastingBenchmark

        benchmark = ForecastingBenchmark(
            backend=backend,
            backend_params=backend_params,
            return_data=return_data,
        )
    else:
        raise ValueError(f"Unsupported benchmark kind: {benchmark_kind}")

    return benchmark._run_validation(task, estimator)


def main():
    """Read a cloudpickled payload from stdin and write results to stdout."""
    import cloudpickle

    payload = cloudpickle.load(sys.stdin.buffer)

    try:
        folds = _run_validation(payload)
        result = {"status": "success", "folds": folds}
    except Exception as exc:
        result = {
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    cloudpickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    main()
