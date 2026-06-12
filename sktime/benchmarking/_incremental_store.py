"""Incremental crash-safe storage for partial benchmark results."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from sktime.benchmarking._benchmarking_dataclasses import ResultObject
from sktime.benchmarking._storage_handlers import JSONStorageHandler

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _atomic_write_text(path: Path, contents: str) -> None:
    """Write text to *path* atomically via a temporary file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(contents)
    tmp_path.replace(path)


def result_key(task_id: str, model_id: str) -> str:
    """Return a stable hash key for a task-model pair."""
    payload = f"{task_id}|{model_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IncrementalResultStore:
    """Crash-safe incremental store for individual benchmark results.

    Each completed task-estimator run is persisted as a pair of files in a
    ``{output_file}.parts/`` directory:

    * ``{hash}.json`` — serialized result (JSON format used by JSONStorageHandler)
    * ``{hash}.done`` — completion marker with content SHA256

    Parameters
    ----------
    output_path : str, Path, or None
        Path to the final benchmark output file. When ``None``, the store is
        disabled (no persistence).
    """

    def __init__(self, output_path: str | Path | None) -> None:
        if output_path is None:
            self.output_path: Path | None = None
            self.parts_dir: Path | None = None
        else:
            self.output_path = Path(output_path)
            self.parts_dir = Path(f"{output_path}.parts")

    def save_result(self, result: ResultObject) -> None:
        """Persist a single completed result incrementally."""
        if self.parts_dir is None:
            return

        self.parts_dir.mkdir(parents=True, exist_ok=True)
        key = result_key(result.task_id, result.model_id)
        result_path = self.parts_dir / f"{key}.json"
        done_path = self.parts_dir / f"{key}.done"

        contents = json.dumps(JSONStorageHandler.serialize_result(result))
        _atomic_write_text(result_path, contents)

        content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
        done_contents = json.dumps(
            {"schema_version": SCHEMA_VERSION, "sha256": content_hash}
        )
        _atomic_write_text(done_path, done_contents)

    def load_results(self) -> list[ResultObject]:
        """Load all valid completed partial results from the parts directory."""
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
        """Return True if the result file is complete and uncorrupted."""
        if not json_path.exists() or not done_path.exists():
            return False
        try:
            with open(json_path, "rb") as file:
                contents = file.read()
            with open(done_path, encoding="utf-8") as file:
                done_data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False

        if done_data.get("schema_version") != SCHEMA_VERSION:
            return False

        content_hash = hashlib.sha256(contents).hexdigest()
        return content_hash == done_data.get("sha256")

    def cleanup(self) -> None:
        """Remove the partial results directory after successful final merge."""
        if self.parts_dir is not None and self.parts_dir.exists():
            shutil.rmtree(self.parts_dir)
