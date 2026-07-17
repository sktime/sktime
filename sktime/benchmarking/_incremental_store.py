"""Internal crash-safe checkpoint store for benchmark runs.

This module provides incremental persistence used during a benchmark run.
It is not part of the public, user-facing storage API and is not a
`BaseStorageHandler` subclass.

Final results are written to a single results file in the user's chosen
format (JSON, CSV, or Parquet) via
`sktime.benchmarking._storage_handlers`. Partial checkpoints are always
stored as JSON fragments in a ``{results_file}.parts/`` directory next
to that file, regardless of the final format. This avoids duplicating
format-specific checkpoint logic for every storage backend.

`IncrementalResultStore` is coordinated by `BenchmarkResultsPersistence`.
Benchmark orchestration code should not import or use it directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from sktime.benchmarking._benchmarking_dataclasses import ResultObject
from sktime.benchmarking._storage_handlers import JSONStorageHandler, _atomic_write_text

logger = logging.getLogger(__name__)


def result_key(task_id: str, model_id: str) -> str:
    """Return a SHA256 hash key for a task-model pair.

    Used as the filename stem for checkpoint files in the ``.parts/``
    directory.

    Parameters
    ----------
    task_id : str
        Task identifier.
    model_id : str
        Model (estimator) identifier.

    Returns
    -------
    str
        Hex-encoded SHA256 digest of ``"{task_id}|{model_id}"``.
    """
    payload = f"{task_id}|{model_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IncrementalResultStore:
    """Crash-safe checkpoint store for individual benchmark results.

    Persists each completed task-estimator run as a pair of files in a
    ``{output_path}.parts/`` directory (where ``output_path`` is the final
    results file path):

    * ``{hash}.json`` — serialized result (uses ``JSONStorageHandler`` format)
    * ``{hash}.done`` — completion marker with a content SHA256 checksum

    A result is only considered valid when both files exist and the checksum
    matches. Incomplete or corrupted entries are skipped on load.

    Parameters
    ----------
    output_path : str, pathlib.Path, or None
        Path to the benchmark results file (e.g. ``"results.csv"``).
        Must refer to a file, not a directory. The checkpoint directory is
        derived as ``f"{output_path}.parts"``. When ``None``, the store is
        disabled and all methods are not performed.

    Attributes
    ----------
    output_path : pathlib.Path or None
        Coerced path to the final results file.
    parts_dir : pathlib.Path or None
        Directory holding incremental checkpoint files (``{output_path}.parts``).

    Notes
    -----
    This is an internal implementation detail, not a final-format storage
    handler. Checkpoints are always JSON regardless of whether the user's
    final results file is CSV, Parquet, etc. See
    `BenchmarkResultsPersistence` for the public persistence interface.
    """

    def __init__(self, output_path: str | Path | None) -> None:
        if output_path is None:
            self.output_path: Path | None = None
            self.parts_dir: Path | None = None
        else:
            self.output_path = Path(output_path)
            self.parts_dir = Path(f"{output_path}.parts")

    def save_result(self, result: ResultObject) -> None:
        """Persist a single completed result as an atomic checkpoint file pair.

        Creates ``{output_path}.parts/`` if needed, then writes the JSON
        payload and ``.done`` marker atomically so a crash mid-write never
        produces a loadable but incomplete result. No operation when
        ``output_path`` is ``None``.

        Parameters
        ----------
        result : ResultObject
            A completed experiment result to checkpoint.
        """
        if self.parts_dir is None:
            return

        self.parts_dir.mkdir(parents=True, exist_ok=True)
        key = result_key(result.task_id, result.model_id)
        result_path = self.parts_dir / f"{key}.json"
        done_path = self.parts_dir / f"{key}.done"

        contents = json.dumps(JSONStorageHandler.serialize_result(result))
        _atomic_write_text(result_path, contents)

        content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
        done_contents = json.dumps({"sha256": content_hash})
        _atomic_write_text(done_path, done_contents)

    def load_results(self) -> list[ResultObject]:
        """Load all valid completed partial results from ``parts_dir``.

        Scans ``{output_path}.parts/`` for ``*.json`` files and returns only
        entries with a matching, checksum-valid ``.done`` marker.

        Returns
        -------
        list of ResultObject
            Valid checkpoint results. Entries missing a ``.done`` file, with
            a checksum mismatch, or with invalid JSON are skipped with a
            warning logged. Returns an empty list when ``output_path`` is
            ``None`` or the parts directory does not exist.
        """
        if self.parts_dir is None or not self.parts_dir.exists():
            return []

        results: list[ResultObject] = []
        for json_path in sorted(self.parts_dir.glob("*.json")):
            key = json_path.stem
            done_path = self.parts_dir / f"{key}.done"
            if not self._is_valid_result(json_path, done_path):
                logger.warning(
                    "Ignoring incomplete or corrupted partial result: %s",
                    json_path.name,
                )
                continue
            try:
                with open(json_path, encoding="utf-8") as file:
                    row = json.load(file)
                results.append(JSONStorageHandler.deserialize_result(row))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid partial result JSON: %s",
                    json_path.name,
                )
                continue
        return results

    @staticmethod
    def _is_valid_result(json_path: Path, done_path: Path) -> bool:
        """Return whether a checkpoint JSON file and its ``.done`` marker match.

        Parameters
        ----------
        json_path : pathlib.Path
            Path to the checkpoint JSON file inside ``.parts/``.
        done_path : pathlib.Path
            Path to the companion ``.done`` marker file.

        Returns
        -------
        bool
            ``True`` if both files exist and the stored SHA256 matches the
            JSON contents.
        """
        if not json_path.exists() or not done_path.exists():
            return False
        try:
            with open(json_path, "rb") as file:
                contents = file.read()
            with open(done_path, encoding="utf-8") as file:
                done_data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False

        content_hash = hashlib.sha256(contents).hexdigest()
        return content_hash == done_data.get("sha256")

    def cleanup(self) -> None:
        """Remove the ``{output_path}.parts/`` checkpoint directory.

        Called after a successful final write to the results file. No
        operation when ``output_path`` is ``None`` or the directory does
        not exist.
        """
        if self.parts_dir is not None and self.parts_dir.exists():
            shutil.rmtree(self.parts_dir)
