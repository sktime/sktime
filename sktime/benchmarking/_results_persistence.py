"""Persistence layer for benchmark results.

This module is the single entry point for all benchmark result I/O.
It coordinates two complementary mechanisms:

1. **Final-format storage** — selected by file extension via the strategy
   pattern in `sktime.benchmarking._storage_handlers` (JSON, CSV,
   Parquet, etc.).
2. **Incremental checkpoints** — crash-safe partial writes managed internally
   by `IncrementalResultStore`.
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
        Path to the final benchmark output file (e.g. ``"results.json"``).
        The file extension determines the storage format. When ``None``,
        persistence is disabled and all methods become no-ops except
        :meth:`load`, which returns an empty list.

    Notes
    -----
    Load priority on `load`:

    1. If the final output file exists, load from it (completed run).
    2. Otherwise, if partial checkpoint files exist, load those (interrupted
       run).
    3. Otherwise, return an empty list (fresh run).

    Incremental checkpoints are always stored as JSON fragments in a
    ``{path}.parts/`` directory, independent of the final output format.
    They are removed automatically by `save_final` after a successful
    final write.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        handler_cls = get_storage_backend(path)
        self._final_handler = handler_cls(path)
        self._incremental_store = IncrementalResultStore(path)

    def load(self) -> list[ResultObject]:
        """Load previously persisted results.

        Returns
        -------
        list of ResultObject
            Loaded results. Returns an empty list when ``path`` is ``None``,
            when no output file or partial checkpoints exist, or when partial
            checkpoints are incomplete or corrupted.

        Notes
        -----
        A completed final output file takes precedence over partial
        checkpoint files. This ensures that a successfully finished run is
        never overwritten by stale incremental artifacts.
        """
        if self.path is None:
            return []

        output_path = Path(self.path)
        if output_path.exists():
            return self._final_handler.load()

        return self._incremental_store.load_results()

    def persist_result(self, result: ResultObject) -> None:
        """Write a single result to incremental checkpoint storage.

        Called after each task-estimator experiment completes so that
        progress survives process crashes. Has no effect when ``path`` is
        ``None``.

        Parameters
        ----------
        result : ResultObject
            A completed experiment result to checkpoint.
        """
        self._incremental_store.save_result(result)

    def save_final(self, results: list[ResultObject]) -> None:
        """Write all results to the final output file and remove checkpoints.

        Parameters
        ----------
        results : list of ResultObject
            Complete in-memory result set to persist.

        Raises
        ------
        RuntimeError
            If the final output file was not created after the save operation.

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
