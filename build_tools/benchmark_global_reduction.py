"""Reproduce global recursive reduction timings, optionally against git revisions.

Example (run from the repository root)::

    python build_tools/benchmark_global_reduction.py --baseline upstream/main \
        --baseline 277635f --series 50 200 800 --repeats 3 --output timings.json

Each baseline supplies only ``_create_fcst_df``. All variants use the same fitted
regressor, dependencies, and remaining prediction code to isolate the helper's
effect. Forecasts must match exactly. Timings are medians after a warm-up; they
are diagnostic measurements, not CI pass/fail thresholds.
Variant order rotates between repetitions to reduce ordering and load bias.
Only use trusted baseline revisions: their helper definitions are executed.
"""

import argparse
import ast
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from sktime.forecasting.compose import _reduce, make_reduction


def _helper_at_revision(revision):
    """Load only the helper definition from an explicitly requested git revision."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Comparing revisions requires git")
    # Explicit developer-selected git refs, passed as arguments without a shell.
    source = subprocess.run(  # noqa: S603
        [
            git,
            "show",
            "--end-of-options",
            f"{revision}:sktime/forecasting/compose/_reduce.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_fcst_df"
    )
    namespace = dict(_reduce.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), revision, "exec"), namespace)
    return namespace["_create_fcst_df"]


def _panel(n_series):
    """Generate the reproducible 60-period panel from issue #11329."""
    rng = np.random.default_rng(0)
    time = np.arange(60)
    values = [
        100 + 10 * np.sin(2 * np.pi * time / 4) + rng.normal(0, 5, 60).cumsum()
        for _ in range(n_series)
    ]
    index = pd.MultiIndex.from_product(
        [[f"s{i}" for i in range(n_series)], time], names=["series", "time"]
    )
    return pd.DataFrame({"y": np.concatenate(values)}, index=index)


def benchmark(baselines, sizes, repeats):
    """Measure prediction and frame-construction costs with numerical checks."""
    revised = _reduce._create_fcst_df
    helpers = {revision: _helper_at_revision(revision) for revision in baselines}
    helpers["working_tree"] = revised
    results = []
    try:
        for n_series in sizes:
            y = _panel(n_series)
            start = perf_counter()
            forecaster = make_reduction(
                LinearRegression(),
                strategy="recursive",
                window_length=8,
                pooling="global",
            ).fit(y)
            fit_time = perf_counter() - start
            expected = None
            samples = {name: [] for name in helpers}
            helper_samples = {name: [] for name in helpers}
            for name, helper in helpers.items():
                _reduce._create_fcst_df = helper
                prediction = forecaster.predict(fh=list(range(1, 9)))
                if expected is None:
                    expected = prediction
                else:
                    pd.testing.assert_frame_equal(
                        prediction, expected, check_exact=True
                    )
            names = list(helpers)
            for repetition in range(repeats):
                offset = repetition % len(names)
                for name in names[offset:] + names[:offset]:
                    helper = helpers[name]
                    _reduce._create_fcst_df = helper
                    start = perf_counter()
                    prediction = forecaster.predict(fh=list(range(1, 9)))
                    samples[name].append(perf_counter() - start)
                    pd.testing.assert_frame_equal(
                        prediction, expected, check_exact=True
                    )
                    start = perf_counter()
                    helper(pd.Index(range(60, 68)), y)
                    helper_samples[name].append(perf_counter() - start)
                print(
                    f"Completed {n_series} series, repeat {repetition + 1}/{repeats}",
                    file=sys.stderr,
                    flush=True,
                )
            for name in names:
                result = {
                    "series": n_series,
                    "variant": name,
                    "fit_seconds": fit_time,
                    "predict_seconds": median(samples[name]),
                    "predict_samples_seconds": samples[name],
                    "frame_seconds": median(helper_samples[name]),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        _reduce._create_fcst_df = revised
    return results


def main():
    """Run the optional developer benchmark without changing checkout files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--series", type=int, nargs="+", default=[50, 200, 800])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.series):
        parser.error("Series counts and repeats must be positive")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "threads": 1,
        "repeats": args.repeats,
        "periods": 60,
        "window_length": 8,
        "horizon": list(range(1, 9)),
    }
    print(json.dumps(environment), flush=True)
    with threadpool_limits(limits=1):
        results = benchmark(args.baseline, args.series, args.repeats)
    if args.output is not None:
        args.output.write_text(
            json.dumps({"environment": environment, "results": results}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
