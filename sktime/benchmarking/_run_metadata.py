"""Run metadata sidecar for benchmark results.

When a benchmark run writes results to a path, a JSON sidecar file is written
next to it recording how the run was produced: sktime version, dependency
versions, system information, a UTC timestamp, the number of tasks and
estimators, and the results path itself.

The sidecar is deliberately kept out of the results file. Benchmark results
are stored in whichever format the results file extension implies (JSON, CSV,
or Parquet) by `sktime.benchmarking._storage_handlers`, and metadata does not
fit the tabular formats without complicating their serialization. Writing a
separate file keeps the result serialization untouched.

The sidecar path appends `.meta.json` to the full results path, so
``results.csv`` yields ``results.csv.meta.json``. This mirrors how
`sktime.benchmarking._results_persistence` derives its ``{path}.parts/``
checkpoint directory, and it is injective: results files that differ only by
extension keep distinct sidecars.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from sktime.benchmarking._storage_handlers import _atomic_write_text
from sktime.utils._maint._show_versions import (
    DEFAULT_DEPS_TO_SHOW,
    _get_deps_info,
    _get_sys_info,
)

METADATA_SUFFIX = ".meta.json"


def get_metadata_path(results_path):
    """Return the metadata sidecar path for a results file path.

    Parameters
    ----------
    results_path : str, pathlib.Path, or None
        Path to the benchmark results file, e.g. ``"results.csv"``.

    Returns
    -------
    pathlib.Path or None
        Path of the metadata sidecar, formed by appending ``.meta.json`` to
        ``results_path``, e.g. ``"results.csv.meta.json"``. ``None`` when
        ``results_path`` is ``None``, in which case no results file is written
        and there is no path to derive a sidecar from.
    """
    if results_path is None:
        return None
    return Path(str(results_path) + METADATA_SUFFIX)


def collect_run_metadata(results_path, n_tasks, n_estimators):
    """Collect metadata describing a benchmark run.

    Parameters
    ----------
    results_path : str, pathlib.Path, or None
        Path the benchmark results were written to.
    n_tasks : int
        Number of tasks registered on the benchmark.
    n_estimators : int
        Number of estimators registered on the benchmark.

    Returns
    -------
    dict
        JSON serializable metadata, with keys:

        * ``sktime_version`` : str, version of the running sktime
        * ``timestamp`` : str, ISO 8601 UTC time the metadata was collected
        * ``n_tasks`` : int, number of registered tasks
        * ``n_estimators`` : int, number of registered estimators
        * ``output_path`` : str or None, the results file path
        * ``sys_info`` : dict, from `_get_sys_info`, python version,
          executable, and platform
        * ``deps_info`` : dict, from `_get_deps_info`, versions of the
          dependencies in ``DEFAULT_DEPS_TO_SHOW``; entries are ``None``
          for packages that are not installed

    Notes
    -----
    ``sktime_version`` is read from ``sktime.__version__``, which describes
    the code actually executing the run. ``deps_info["sktime"]`` reports the
    installed distribution metadata instead, which can differ from it in a
    development or editable install.
    """
    import sktime

    return {
        "sktime_version": sktime.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_tasks": n_tasks,
        "n_estimators": n_estimators,
        "output_path": None if results_path is None else str(results_path),
        "sys_info": _get_sys_info(),
        "deps_info": _get_deps_info(deps=DEFAULT_DEPS_TO_SHOW),
    }


def write_run_metadata(results_path, n_tasks, n_estimators):
    """Write the metadata sidecar next to a benchmark results file.

    Parameters
    ----------
    results_path : str, pathlib.Path, or None
        Path the benchmark results were written to. When ``None``, no results
        file exists and nothing is written.
    n_tasks : int
        Number of tasks registered on the benchmark.
    n_estimators : int
        Number of estimators registered on the benchmark.

    Returns
    -------
    pathlib.Path or None
        Path of the sidecar written. ``None`` if ``results_path`` is ``None``,
        or if the sidecar could not be written, in which case no file exists at
        the sidecar path and a warning has been raised.

    Warns
    -----
    UserWarning
        If writing the sidecar fails with an ``OSError``, e.g. because the
        directory is not writable or the disk is full.

    Notes
    -----
    The sidecar is written atomically, via a temporary file, so an interrupted
    write leaves any previous metadata file intact. An existing sidecar for the
    same results path is overwritten, so it describes the most recent run.

    A failure to write the sidecar warns rather than raises. Metadata is
    written after the benchmark results have already been persisted, so
    raising here would discard a completed run over the loss of a descriptive
    file that does not affect the results themselves.
    """
    metadata_path = get_metadata_path(results_path)
    if metadata_path is None:
        return None

    metadata = collect_run_metadata(results_path, n_tasks, n_estimators)
    try:
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
    except OSError as exc:
        warnings.warn(
            f"Failed to write benchmark run metadata to {metadata_path}: "
            f"{type(exc).__name__}: {exc}. "
            "Benchmark results were saved and are unaffected.",
            stacklevel=2,
        )
        return None
    return metadata_path
