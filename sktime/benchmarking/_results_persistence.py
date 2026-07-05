"""Persistence layer for benchmark results.

This module is the single entry point for all benchmark result I/O.
It coordinates two complementary mechanisms:

1. **Final-format storage** — a single results file at ``path``,
   with format selected by file extension via
   `sktime.benchmarking._storage_handlers` (JSON, CSV, Parquet).
2. **Incremental checkpoints** — crash-safe partial writes in a
   ``{path}.parts/`` directory, managed internally by
   `IncrementalResultStore`.
"""

from __future__ import annotations

from pathlib import Path

from sktime.benchmarking._benchmarking_dataclasses import ResultObject
from sktime.benchmarking._incremental_store import IncrementalResultStore
from sktime.benchmarking._storage_handlers import get_storage_backend


class BenchmarkResultsPersistence:
    """Facade for loading, checkpointing, and saving benchmark results.

    This class owns all persistence concerns for a benchmark run:

    * selecting the final-format storage backend from the output path
    * loading previously saved results (final file or partial checkpoints)
    * writing crash-safe incremental checkpoints after each experiment
    * merging in-memory results to the final output file and cleaning up
      partial artifacts

    Parameters
    ----------
    path : str or None
        Path to the benchmark results file (e.g. ``"results.csv"``).
        Must refer to a file, not a directory; the file extension determines
        the final storage format (``.json``, ``.csv``, or ``.parquet``).
        When ``None``, persistence is disabled: `load` returns an
        empty list and `persist_result` / `save_final` are
        not performed.

    Notes
    -----
    Load priority in `load`:

    1. If the final results file at ``path`` exists, load from it
       (completed run).
    2. Otherwise, if checkpoint files exist in ``{path}.parts/``, load
       those (interrupted run).
    3. Otherwise, return an empty list (fresh run).

    Incremental checkpoints are stored as JSON fragments in a
    ``{path}.parts/`` **directory** (derived from the results file path),
    independent of the final output format. That directory is removed
    automatically by `save_final` after a successful final write.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        handler_cls = get_storage_backend(path)
        self._final_handler = handler_cls(path)
        self._incremental_store = IncrementalResultStore(path)

    def load(self) -> list[ResultObject]:
        """Load previously persisted results from disk.

        Reads from the final results file at ``path`` when it exists;
        otherwise falls back to valid checkpoints in ``{path}.parts/``.

        Returns
        -------
        list of ResultObject
            Loaded results. Returns an empty list when ``path`` is ``None``,
            when neither the results file nor checkpoint directory exists,
            or when checkpoints are incomplete or corrupted.

        Notes
        -----
        A completed final results file takes precedence over partial
        checkpoint files.
        """
        if self.path is None:
            return []

        output_path = Path(self.path)
        if output_path.exists():
            return self._final_handler.load()

        return self._incremental_store.load_results()

    def persist_result(self, result: ResultObject) -> None:
        """Write a single result to incremental checkpoint storage.

        Appends one JSON checkpoint file pair under ``{path}.parts/`` after
        each task-estimator experiment completes so progress survives process
        crashes. Has no effect when ``path`` is ``None``.

        Parameters
        ----------
        result : ResultObject
            A completed experiment result to checkpoint.
        """
        self._incremental_store.save_result(result)

    def save_final(self, results: list[ResultObject]) -> None:
        """Write all results to the final file at ``path`` and remove checkpoints.

        Overwrites the results file in the format implied by its extension.
        No operation when ``path`` is ``None``.

        Parameters
        ----------
        results : list of ResultObject
            Complete in-memory result set to persist.

        Raises
        ------
        RuntimeError
            If the results file at ``path`` was not created after the save
            operation.

        Notes
        -----
        On success, the ``{path}.parts/`` checkpoint directory is deleted.
        """
        if self.path is None:
            return

        self._final_handler.save(results)

        if not Path(self.path).exists():
            raise RuntimeError(f"Failed to save benchmark results to {self.path}")

        self._incremental_store.cleanup()
