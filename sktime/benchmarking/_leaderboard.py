"""Leaderboard table for benchmark results.

Turns the flat results table from ``BaseBenchmark.run`` /
``BenchmarkingResults.to_df`` into one row per model. Each scorer has its own
``<scorer>_rank``, averaged across the selected tasks.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import re
import warnings
from functools import lru_cache
from os import PathLike

import pandas as pd

# column suffix used by ``ResultObject.to_dataframe`` for aggregated scores
_MEAN_SUFFIX = "_mean"

# aggregated columns that are timings, not estimation metrics
_NON_METRIC_BASES = frozenset({"fit_time", "pred_time", "runtime"})

# ``[dataset=<name>]`` field of a generated task id
_DATASET_FIELD = re.compile(r"\[dataset=(.*?)]")


def _build_leaderboard(
    results,
    dataset=None,
    scorers=None,
    lower_is_better=None,
    scorer_directions=None,
):
    """Build a leaderboard dataframe from flat benchmark results.

    Parameters
    ----------
    results : pandas.DataFrame
        Flat results table from ``BaseBenchmark.run`` or
        ``BenchmarkingResults.to_df``. One row per task and model.
    dataset : str or list of str, optional (default=None)
        Tasks to rank on. ``None`` uses every task. A string, or each entry
        of a list, may be a dataset name (the ``[dataset=...]`` field of a
        task id) or a full ``validation_id``.
    scorers : str, list, metric, or list of metrics, optional (default=None)
        Scorers to rank on. ``None`` uses every scorer column in ``results``.
        A metric object or class is identified by its ``name`` or ``__name__``.
    lower_is_better : bool or dict of str to bool, optional (default=None)
        Direction of each scorer. ``True`` gives rank 1 to the lowest score.
        A bool applies to every selected scorer. A dict maps scorer name to
        direction and overrides inference for those scorers. When ``None``,
        the direction is taken from ``scorer_directions``, then from the
        sktime metric registry, then from the scorer name.
    scorer_directions : dict of str to bool, optional (default=None)
        Pre-resolved ``lower_is_better`` flags, usually taken from the metric
        objects registered on a benchmark. Explicit ``lower_is_better`` wins.

    Returns
    -------
    pandas.DataFrame
        One row per model. Ranks from different metrics are not averaged.
        Rows are ordered by the first scorer's rank.

        * ``model_id`` — estimator identifier.
        * ``<scorer>_rank`` — that scorer's rank, averaged across the selected
          tasks. Within a task, ties use the average of the tied positions.
          One column per scorer, named with the scorer.
        * one column per scorer — mean score across the selected tasks. Each
          task's score is already the mean over folds.
        * ``fit_time_s``, ``predict_time_s`` — mean fit and predict time in
          seconds across the selected tasks. Present when the results contain
          ``fit_time_mean`` and ``pred_time_mean``.

        Column order is ``model_id``, then each ``<scorer>_rank``, then each
        scorer's score, then ``fit_time_s`` and ``predict_time_s``.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If no results, tasks, or scorers match the request, or if a model is
        missing a score on a selected task.
    """
    frame = _validate_results(results)
    frame = _filter_datasets(frame, dataset)
    selected = _select_scorers(frame, scorers)
    directions = _resolve_directions(selected, lower_is_better, scorer_directions)

    per_scorer_ranks = {}
    per_scorer_scores = {}
    for scorer in selected:
        ranks, scores = _rank_scorer(frame, scorer, lower_is_better=directions[scorer])
        per_scorer_ranks[scorer] = ranks
        per_scorer_scores[scorer] = scores

    # Each scorer keeps its own rank, named ``<scorer>_rank``. Ranks of
    # different metrics are not collapsed into one column.
    board = pd.DataFrame(index=per_scorer_ranks[selected[0]].index)
    board.index.name = "model_id"
    _attach_times(board, frame)
    for scorer in selected:
        board[f"{scorer}_rank"] = per_scorer_ranks[scorer]
        board[scorer] = per_scorer_scores[scorer]

    board = board.reset_index()
    board = board.sort_values(
        [f"{selected[0]}_rank", "model_id"], kind="mergesort"
    ).reset_index(drop=True)
    return _order_columns(board, selected)


def _order_columns(board, scorers):
    """Place model id, then ranks, then scores, then timings."""
    rank_columns = [f"{scorer}_rank" for scorer in scorers]
    columns = ["model_id", *rank_columns, *list(scorers)]
    for timing in ("fit_time_s", "predict_time_s"):
        if timing in board.columns:
            columns.append(timing)
    return board[columns]


def _validate_results(results):
    """Return a results table with the columns a leaderboard requires."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Benchmark results must be a pandas DataFrame, "
            f"got {type(results).__name__}."
        )
    if results.empty:
        raise ValueError(
            "No benchmark results available to build a leaderboard. "
            "Run the benchmark, or load a results file that contains results."
        )
    missing = {"validation_id", "model_id"} - set(results.columns)
    if missing:
        raise ValueError(
            "Results are missing required columns: "
            f"{sorted(missing)}. Expected the table returned by "
            "`BaseBenchmark.run` or `BenchmarkingResults.to_df`."
        )
    duplicated = results.duplicated(subset=["validation_id", "model_id"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Results contain duplicate (validation_id, model_id) rows. "
            "A leaderboard needs one row per task and model."
        )
    return results


def _dataset_label(validation_id):
    """Return the dataset name embedded in a generated task id, or None."""
    match = _DATASET_FIELD.search(str(validation_id))
    if match:
        return match.group(1)
    return None


def _filter_datasets(results, dataset):
    """Keep rows whose task id or dataset name was requested."""
    if dataset is None:
        return results

    if isinstance(dataset, str):
        requested = [dataset]
    elif isinstance(dataset, (list, tuple)):
        if not dataset or not all(isinstance(item, str) for item in dataset):
            raise TypeError(
                "`dataset` must be a string or a non-empty list of strings."
            )
        requested = list(dataset)
    else:
        raise TypeError(
            "`dataset` must be a string or a list of strings, "
            f"got {type(dataset).__name__}."
        )

    labels = results["validation_id"].map(_dataset_label)
    selected = results["validation_id"].isin(requested) | labels.isin(requested)
    filtered = results.loc[selected]
    if filtered.empty:
        available = sorted(
            set(labels.dropna().tolist()) | set(results["validation_id"].astype(str))
        )
        raise ValueError(
            f"No results matched dataset={requested!r}. "
            f"Available dataset names and task ids: {available}."
        )
    return filtered


def _metric_names(results):
    """Scorer names that have an aggregated ``<name>_mean`` column."""
    names = []
    for column in results.columns:
        if not column.endswith(_MEAN_SUFFIX):
            continue
        base = column[: -len(_MEAN_SUFFIX)]
        if base not in _NON_METRIC_BASES:
            names.append(base)
    return list(dict.fromkeys(names))


def _scorer_name(scorer):
    """Name of a metric object or callable, as stored in the results."""
    name = getattr(scorer, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(scorer, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return None


def _select_scorers(results, scorers):
    """Resolve the requested scorers against columns of ``results``."""
    available = _metric_names(results)
    if not available:
        raise ValueError(
            "No '<scorer>_mean' columns found in results. "
            f"Available columns: {list(results.columns)}."
        )
    if scorers is None:
        return available

    requested = _coerce_scorer_names(scorers)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"Scorer(s) {missing} are not in the results. "
            f"Available scorers: {available}."
        )
    return requested


def _coerce_scorer_names(scorers):
    """Return scorer names from a name, a metric, or a list of either."""
    if isinstance(scorers, (list, tuple)):
        items = list(scorers)
    else:
        items = [scorers]
    if not items:
        raise ValueError("`scorers` must contain at least one scorer.")

    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
            continue
        name = _scorer_name(item)
        if name is None:
            raise TypeError(
                "Each entry of `scorers` must be a scorer name, or a metric "
                "with a `name` or `__name__`. "
                f"Got {type(item).__name__}."
            )
        names.append(name)
    return names


def _resolve_directions(scorers, lower_is_better, scorer_directions):
    """Map each scorer name to whether lower scores rank higher."""
    if lower_is_better is not None and not isinstance(lower_is_better, (bool, dict)):
        raise TypeError(
            "`lower_is_better` must be a bool or a dict of scorer name to bool, "
            f"got {type(lower_is_better).__name__}."
        )
    if isinstance(lower_is_better, dict):
        unknown = [key for key in lower_is_better if key not in scorers]
        if unknown:
            raise ValueError(
                f"`lower_is_better` refers to unknown scorer(s) {unknown}. "
                f"Selected scorers: {list(scorers)}."
            )
        invalid = [
            key for key, value in lower_is_better.items() if not isinstance(value, bool)
        ]
        if invalid:
            raise TypeError(
                "`lower_is_better` values must be bool. "
                f"Entries that are not bool: {invalid}."
            )

    known = dict(scorer_directions or {})
    resolved = {}
    unresolved = []
    for scorer in scorers:
        if isinstance(lower_is_better, bool):
            resolved[scorer] = lower_is_better
            continue
        if isinstance(lower_is_better, dict) and scorer in lower_is_better:
            resolved[scorer] = lower_is_better[scorer]
            continue
        if scorer in known:
            resolved[scorer] = bool(known[scorer])
            continue
        inferred = _infer_lower_is_better(scorer)
        if inferred is None:
            unresolved.append(scorer)
            resolved[scorer] = True
        else:
            resolved[scorer] = inferred

    if unresolved:
        warnings.warn(
            "Could not determine whether lower values are better for "
            f"{unresolved}. Assuming lower is better. Pass `lower_is_better` "
            "to set the direction, for example "
            '`lower_is_better={"my_metric": False}`.',
            UserWarning,
            stacklevel=4,
        )
    return resolved


def _infer_lower_is_better(scorer_name):
    """Infer a scorer direction from its name or the metric registry.

    Returns
    -------
    bool or None
        ``True`` when lower scores are better. ``None`` when no rule applies
        and the caller should fall back and warn.
    """
    lower, specific = _direction_from_name(scorer_name)
    if specific:
        return lower

    registered = _metric_class_directions().get(scorer_name)
    if registered is not None:
        return registered
    return None


def _direction_from_name(scorer):
    """Return ``(lower_is_better, specific)`` from a name or callable.

    ``specific`` is ``False`` when the only conclusion is the generic
    "assume lower is better" fallback. Callers then try the metric registry
    before accepting that fallback.
    """
    from sktime.utils.sklearn._scoring import _guess_sign_of_sklmetric

    if not isinstance(scorer, str) and hasattr(scorer, "greater_is_better"):
        return (not bool(scorer.greater_is_better)), True

    name = scorer if isinstance(scorer, str) else getattr(scorer, "__name__", None)
    if not isinstance(name, str):
        return True, False

    if isinstance(scorer, str):

        def named_scorer(*args, **kwargs):
            return None

        named_scorer.__name__ = name
        scorer = named_scorer

    sign = _guess_sign_of_sklmetric(scorer)
    specific = (
        sign > 0
        or name.endswith(("_score", "_loss", "_deviance", "_error"))
        or name == "class_likelihood_ratios"
    )
    return sign < 0, specific


@lru_cache(maxsize=1)
def _metric_class_directions():
    """Map sktime metric class names to their ``lower_is_better`` tag."""
    from sktime.registry import all_estimators

    entries = all_estimators(
        estimator_types="metric",
        return_tags=["lower_is_better"],
    )
    directions = {}
    for entry in entries:
        name = entry[0]
        lower = entry[-1]
        if isinstance(lower, bool):
            directions[name] = lower
    return directions


def _directions_from_scorers(scorers):
    """Read ``lower_is_better`` from metric objects and known callables."""
    directions = {}
    for scorer in scorers:
        name = _scorer_name(scorer)
        if name is None:
            continue
        if hasattr(scorer, "get_tag"):
            try:
                directions[name] = bool(scorer.get_tag("lower_is_better"))
            except (AttributeError, ValueError):
                pass
            else:
                continue
        lower, specific = _direction_from_name(scorer)
        if specific:
            directions[name] = lower
    return directions


def _rank_scorer(results, scorer, lower_is_better):
    """Rank models on one scorer and return average rank and mean score.

    Ranks are computed within each task, then averaged across tasks with
    equal weight. Rank 1 is the best score.
    """
    column = f"{scorer}{_MEAN_SUFFIX}"
    if column not in results.columns:
        raise ValueError(
            f"Column '{column}' not found in results. "
            f"Available columns: {list(results.columns)}."
        )
    if not pd.api.types.is_numeric_dtype(results[column]):
        raise ValueError(
            f"Scorer '{scorer}' values in '{column}' are not numeric, "
            "so they cannot be ranked."
        )

    scores = results.pivot(index="validation_id", columns="model_id", values=column)
    _check_complete(scores, scorer)
    ranks = scores.rank(axis=1, method="average", ascending=lower_is_better)
    return ranks.mean(axis=0), scores.mean(axis=0)


def _check_complete(scores, scorer):
    """Raise if any model is missing a score on any selected task."""
    if not scores.isna().to_numpy().any():
        return
    missing = [
        (str(model), str(task))
        for task in scores.index
        for model in scores.columns
        if pd.isna(scores.at[task, model])
    ]
    preview = missing[:10]
    suffix = "" if len(missing) <= 10 else f" (and {len(missing) - 10} more)"
    raise ValueError(
        f"Scorer '{scorer}' is missing scores for {len(missing)} "
        f"(model_id, validation_id) pair(s): {preview}{suffix}. "
        "Every model needs a score on every selected task. Restrict "
        "`dataset` or `scorers`, or drop the incomplete rows from the "
        "results, before building the leaderboard."
    )


def _attach_times(board, results):
    """Add mean fit and predict time, in seconds, when the columns exist."""
    renamed = (("fit_time_mean", "fit_time_s"), ("pred_time_mean", "predict_time_s"))
    for source, target in renamed:
        if source not in results.columns:
            continue
        if not pd.api.types.is_numeric_dtype(results[source]):
            continue
        board[target] = results.groupby("model_id")[source].mean()


def _coerce_results(results):
    """Return a flat results dataframe from a dataframe or a file path."""
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, (str, PathLike)):
        from sktime.benchmarking.benchmarks import BenchmarkingResults

        return BenchmarkingResults(path=str(results)).to_df()
    raise TypeError(
        "`results` must be a pandas DataFrame or a path to a benchmark "
        f"results file, got {type(results).__name__}."
    )
