"""Benchmarking for detection estimators."""

from collections.abc import Callable

import pandas as pd

from sktime.benchmarking._benchmarking_dataclasses import FoldResults, TaskObject
from sktime.benchmarking.benchmarks import BaseBenchmark, _get_dataset_name
from sktime.detection.base import BaseDetector

__author__ = ["yash-sangwan"]
__all__ = ["DetectionBenchmark"]


def _leave_one_series_out(X, y=None):
    """Split a panel into one live series, and the panel of all other series.

    One fold per time series in ``X``. The series of the fold is the live
    series, and the other series are the panel the detector pretrains on, so
    a detector never pretrains on the series it is later scored on.

    Parameters
    ----------
    X : pd.DataFrame with row MultiIndex (instance, time)
        Panel of time series. The instance can have more than one level,
        in which case all levels but the last are the instance.
    y : pd.DataFrame, optional
        Known events in ``X``, one row per event, with an ``"ilocs"`` column
        and row ``MultiIndex`` ``(instance, event_no)``, as ``pretrain``
        expects. The ``"ilocs"`` of an event are offsets into its own series.

    Yields
    ------
    instance : label of the live series in ``X``
    X_pretrain : pd.DataFrame
        Panel of all series except the live one.
    y_pretrain : pd.DataFrame, or None
        Known events of the series in ``X_pretrain``. None if ``y`` is None.
    X_live : pd.DataFrame
        The live series alone, with the instance levels dropped, so it is a
        single time series that ``fit`` and ``predict`` accept.
    y_live : pd.DataFrame, or None
        Known events of the live series, with the instance levels dropped.
        None if ``y`` is None.
    """
    instances = X.index.droplevel(-1)
    instance_levels = list(range(X.index.nlevels - 1))

    for instance in instances.unique():
        is_live = instances.isin([instance])

        X_pretrain = X[~is_live]
        X_live = X[is_live].droplevel(instance_levels)

        if y is None:
            y_pretrain = None
            y_live = None
        else:
            y_is_live = y.index.droplevel(-1).isin([instance])
            y_pretrain = y[~y_is_live]
            y_live = y[y_is_live].droplevel(list(range(y.index.nlevels - 1)))

        yield instance, X_pretrain, y_pretrain, X_live, y_live


def _clone_unpretrained(estimator):
    """Return a clone of the estimator, and check it has not been pretrained.

    Every fold must start from a detector that has learnt nothing, so that
    ``pretrain`` runs ``_pretrain`` and not ``_pretrain_update``. The clone
    plugin of ``BaseDetector`` carries pretrained attributes over to clones,
    so a detector that was already pretrained would silently keep what it
    learnt elsewhere.

    Parameters
    ----------
    estimator : BaseDetector
        Detector registered with the benchmark, never pretrained or fitted
        by the benchmark itself.

    Returns
    -------
    BaseDetector
        Clone of ``estimator``, in state ``"new"``.

    Raises
    ------
    ValueError
        If the clone is not in state ``"new"``.
    """
    detector = estimator.clone()

    if detector.state != "new":
        raise ValueError(
            f"{type(estimator).__name__} added to DetectionBenchmark is in state "
            f"{detector.state!r}, but a detector must be in state 'new' when the "
            "benchmark pretrains it. Add a detector that has not been pretrained "
            "or fitted, so that no state can leak between folds."
        )

    return detector


def _replay_live(detector, X_live, warmup, chunk_size):
    """Fit the detector on a warm-up prefix, then replay the rest in chunks.

    The live series is replayed as it would arrive in deployment. The
    detector is fitted on the first ``warmup`` points, then the rest of the
    series is passed to ``update_predict`` in chunks of ``chunk_size``
    points, without known events.

    ``update_predict`` predicts on the chunk it is passed, so the alarms it
    returns are positions inside that chunk. Every alarm of a chunk is
    credited at the last point of the chunk, because in a live replay the
    detector could not have raised it before the whole chunk had arrived.
    Several alarms in one chunk therefore collapse into one, and with
    ``chunk_size=1`` the mapping is the identity.

    Alarms raised on the warm-up prefix are not kept: the warm-up is the
    data the detector is fitted on, so it is not replayed.

    Parameters
    ----------
    detector : BaseDetector
        Detector to replay the series with, already pretrained.
    X_live : pd.DataFrame
        The live series, a single time series.
    warmup : int
        Number of points at the start of ``X_live`` used to fit the detector.
        These points are not replayed.
    chunk_size : int
        Number of points passed to ``update_predict`` at a time.

    Returns
    -------
    pd.DataFrame with a single ``"ilocs"`` column and a RangeIndex
        Alarms of the replay, as ``iloc`` references into ``X_live``, in
        increasing order. All positions are at or after ``warmup``.
    """
    n_timepoints = len(X_live)
    warmup = min(warmup, n_timepoints)

    detector.fit(X_live.iloc[:warmup])

    alarms = []

    for start in range(warmup, n_timepoints, chunk_size):
        end = min(start + chunk_size, n_timepoints)

        # no known events are passed, the replay is unlabelled
        y_chunk = detector.update_predict(X_live.iloc[start:end])

        # every alarm of the chunk is credited at the end of the chunk
        if len(y_chunk) > 0:
            alarms.append(end - 1)

    return pd.DataFrame(
        {"ilocs": pd.Series(alarms, dtype="int64")},
        index=pd.RangeIndex(len(alarms)),
    )


def _events_after_warmup(y_live, warmup):
    """Return the known events a replay of the live series could alarm on.

    Events inside the warm-up prefix are dropped. The warm-up is the data the
    detector is fitted on and is not replayed, so no alarm can be raised on
    it, and a detector must not be scored on those events.

    Parameters
    ----------
    y_live : pd.DataFrame, or None
        Known events of the live series, with an ``"ilocs"`` column whose
        entries are offsets into that series.
    warmup : int
        Number of points at the start of the live series used to fit.

    Returns
    -------
    pd.DataFrame with an ``"ilocs"`` column and a RangeIndex, or None
        Events at or after ``warmup``. None if ``y_live`` is None.
    """
    if y_live is None:
        return None

    return y_live[y_live["ilocs"] >= warmup].reset_index(drop=True)


def _scorer_names(scorers):
    """Return one column name per scorer, from the class name of the scorer.

    Two scorers of the same class, for instance the same metric with
    different offsets, would otherwise share a name and overwrite each other
    in the results, so repeats get a numbered suffix.

    Parameters
    ----------
    scorers : list of BaseDetectionMetric

    Returns
    -------
    list of str, of the same length as ``scorers``
    """
    names = []

    for scorer in scorers:
        base = type(scorer).__name__
        name = base
        suffix = 2
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)

    return names


class DetectionBenchmark(BaseBenchmark):
    """Detection benchmark.

    Run a series of detectors against a series of tasks, defined via a panel
    of time series and their known events, and return the results.

    Each task is evaluated leave-one-series-out: every series of the panel is
    the live series of one fold, and the detector pretrains on the other
    series of that panel only. The detector of a fold is always a fresh clone
    of the detector registered with the benchmark, so nothing learnt in one
    fold can reach the next one.

    Within a fold, the live series is replayed as it would arrive in
    deployment: the detector is fitted on a warm-up prefix, and the rest of
    the series is passed to ``update_predict`` in chunks, without known
    events. Every alarm of a chunk is credited at the last point of that
    chunk, see ``_replay_live``.

    The alarms of a fold are scored once, at the end of the replay, against
    the known events of that live series, with the detection metrics passed
    as ``scorers``. Known events inside the warm-up are not scored.

    Parameters
    ----------
    id_format: str, optional (default=None)
        A regex used to enforce task/estimator ID to match a certain format.

    return_data : bool, optional (default=False)
        Whether to return the data in the results.

    warmup : int, optional (default=1)
        Number of points at the start of a live series used to fit the
        detector. These points are not replayed, and alarms are never
        reported on them. Must be at least 1.

    chunk_size : int, optional (default=1)
        Number of points passed to ``update_predict`` at a time. Alarms of a
        chunk are credited at the last point of the chunk, so ``1`` reports
        the alarm positions unchanged. Must be at least 1.
    """

    def __init__(
        self,
        id_format: str | None = None,
        backend=None,
        backend_params=None,
        return_data=False,
        warmup: int = 1,
        chunk_size: int = 1,
    ):
        super().__init__(
            id_format=id_format,
            backend=backend,
            backend_params=backend_params,
            return_data=return_data,
        )

        if warmup < 1:
            raise ValueError(f"warmup must be at least 1, but found {warmup}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, but found {chunk_size}")

        self.warmup = warmup
        self.chunk_size = chunk_size

    def _add_estimator(
        self,
        estimator: BaseDetector,
        estimator_id: str | None = None,
    ):
        """Register a single detector to the benchmark.

        Parameters
        ----------
        estimator : BaseDetector
            A single initialised detector to add to the benchmark.

        estimator_id : str, optional (default=None)
            Identifier for the detector. If none given, the class name is
            used.

        Raises
        ------
        TypeError
            If the estimator is not a detector.
        """
        if not isinstance(estimator, BaseDetector):
            raise TypeError(
                "DetectionBenchmark benchmarks detectors, but "
                f"{type(estimator).__name__} is not one. Add an object that "
                "inherits from BaseDetector, for instance from sktime.detection."
            )

        super()._add_estimator(estimator, estimator_id)

    def add_task(
        self,
        dataset_loader: Callable | tuple,
        scorers: list | None = None,
        task_id: str | None = None,
    ):
        """Register a detection task to the benchmark.

        Parameters
        ----------
        dataset_loader : Callable or tuple
            The panel to benchmark on. Can be

            - a tuple ``(X, y)``, of a panel of time series and their known
              events. ``X`` has a row ``MultiIndex`` ``(instance, time)``,
              and ``y`` a row ``MultiIndex`` ``(instance, event_no)`` with an
              ``"ilocs"`` column, whose entries are offsets into their own
              series.
            - a single panel ``X``, if there are no known events.
            - a function or dataset object returning either of the above.

        scorers : list of BaseDetectionMetric, optional (default=None)
            Detection metrics the alarms of a fold are scored with, for
            instance ``EventTPR``, ``MeanDetectionOffset`` or
            ``FalseAlarmRate``. Each is called once per live series, with the
            known events of that series, the alarms of the replay, and the
            live series itself. If none given, nothing is scored.

        task_id : str, optional (default=None)
            Identifier for the benchmark task. If none given, it is derived
            from the dataset name.

        Returns
        -------
        None
        """
        if task_id is None:
            task_id = (
                f"[dataset={_get_dataset_name(dataset_loader)}]"
                + "_[split=leave_one_series_out]"
            )

        # the split is leave-one-series-out, so there is no cv_splitter
        self._add_task(
            task_id,
            TaskObject(
                data=dataset_loader,
                cv_splitter=None,
                scorers=list(scorers) if scorers is not None else [],
            ),
        )

    def _run_validation(self, task: TaskObject, estimator: BaseDetector):
        """Pretrain, replay and score the detector once per held-out series.

        One fold per series of the panel. In each fold, the detector is a
        fresh clone of the registered detector, pretrained on the series of
        the panel other than the live one. The live series is then replayed,
        see ``_replay_live``, and its alarms are scored with the scorers of
        the task.

        Parameters
        ----------
        task : TaskObject
            Task registered via ``add_task``.
        estimator : BaseDetector
            Detector registered via ``add_estimator``. Not modified.

        Returns
        -------
        dict of int to FoldResults
            One entry per fold, in the order of the series in the panel.
            Scores are keyed by the class name of the scorer, and empty if
            the task has no scorers. If ``return_data``, the alarms of the
            replay and the scored known events of the live series are
            returned as ``predictions`` and ``ground_truth``.

        Raises
        ------
        ValueError
            If the task has scorers, but no known events to score against.
        """
        data = task.get_y_X("detection")
        X = data["X"]
        y = data["y"]

        scorers = task.scorers
        names = _scorer_names(scorers)

        if len(scorers) > 0 and y is None:
            raise ValueError(
                "DetectionBenchmark scorers need the known events of the panel "
                "to score alarms against, but the task was registered without "
                "them. Pass the data of the task as a tuple (X, y)."
            )

        folds = {}

        for i, split in enumerate(_leave_one_series_out(X, y)):
            _, X_pretrain, y_pretrain, X_live, y_live = split

            detector = _clone_unpretrained(estimator)
            detector.pretrain(X_pretrain, y_pretrain)

            y_pred = _replay_live(detector, X_live, self.warmup, self.chunk_size)

            # the warm-up is not replayed, so its events are not scored
            y_true = _events_after_warmup(y_live, self.warmup)

            scores = {
                name: scorer(y_true=y_true, y_pred=y_pred, X=X_live)
                for name, scorer in zip(names, scorers)
            }

            if self.return_data:
                folds[i] = FoldResults(
                    scores=scores, ground_truth=y_true, predictions=y_pred
                )
            else:
                folds[i] = FoldResults(scores=scores)

        return folds
