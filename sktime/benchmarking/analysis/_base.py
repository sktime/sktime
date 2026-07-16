"""Base class for post-hoc benchmark evaluators (strategy pattern)."""

__all__ = ["BaseBenchmarkAnalyzer"]

import pandas as pd

from sktime.base import BaseObject

# column suffix used by the v2 ``BaseBenchmark.run()`` output for aggregated
# per-(model, task) scores, see ``ResultObject.to_dataframe``.
_MEAN_SUFFIX = "_mean"

# aggregated columns in the v2 output that are not estimation metrics and must
# never be treated as a candidate metric when inferring ``metric``.
_NON_METRIC_BASES = frozenset({"fit_time", "pred_time", "runtime"})


class BaseBenchmarkAnalyzer(BaseObject):
    """Base class for post-hoc statistical benchmark analyzers.

    Benchmark analyzers consume the flat results table produced by the v2
    benchmarking framework (``BaseBenchmark.run()`` /
    ``ResultObject.to_dataframe``) and compute a post-hoc statistical analysis
    (ranking, omnibus / pairwise significance tests, critical-difference
    diagrams).

    The class is **stateless**: ``evaluate`` does not mutate the instance, so a
    single evaluator can be reused across multiple result tables.

    Concrete evaluators implement ``_evaluate(scores)``, where ``scores`` is a
    ``(n_datasets, n_estimators)`` pandas DataFrame (index = ``validation_id``,
    columns = ``model_id``) of the per-task mean score.

    Parameters
    ----------
    metric : str, optional (default=None)
        Name of the metric to analyse, i.e. the ``{metric}`` whose
        ``{metric}_mean`` column is used from the results table. If ``None``,
        the metric is inferred when exactly one metric column is present;
        otherwise a ``ValueError`` is raised.
    lower_is_better : bool, optional (default=True)
        Whether lower metric values indicate better performance (e.g. error
        metrics). Controls ranking direction and the ``is_errors`` semantics of
        the critical-difference diagram.
    """

    _tags = {
        "object_type": "benchmark-analyzer",
        "authors": ["viktorkaz", "mloning", "Aaron Bostrom"],
        "python_dependencies": None,
        "property:analyzer_type": None,
    }

    def __init__(self, metric=None, lower_is_better=True):
        self.metric = metric
        self.lower_is_better = lower_is_better
        super().__init__()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def evaluate(self, results):
        """Run the post-hoc analysis on benchmark results.

        Parameters
        ----------
        results : pandas.DataFrame or str or pathlib.Path
            Either the flat results ``DataFrame`` returned by
            ``BaseBenchmark.run()``, or a path to a result artifact written by a
            storage handler (``.csv`` / ``.json`` / ``.parquet``).

        Returns
        -------
        pandas.DataFrame
            The analysis result. Shape and columns depend on the concrete
            evaluator (see the subclass docstring).
        """
        scores = self._coerce_to_score_matrix(results)
        return self._evaluate(scores)

    # ------------------------------------------------------------------ #
    # strategy hook (abstract)
    # ------------------------------------------------------------------ #
    def _evaluate(self, scores):
        """Compute the post-hoc analysis.

        Parameters
        ----------
        scores : pandas.DataFrame
            ``(n_datasets, n_estimators)`` matrix, index = ``validation_id``,
            columns = ``model_id``.

        Returns
        -------
        pandas.DataFrame
        """
        raise NotImplementedError("abstract method")

    # ------------------------------------------------------------------ #
    # shared adapter: v2 results -> (n_datasets, n_estimators) score matrix
    # ------------------------------------------------------------------ #
    def _coerce_to_score_matrix(self, results):
        """Coerce v2 benchmark output to a ``(n_datasets, n_estimators)`` matrix.

        Parameters
        ----------
        results : pandas.DataFrame or str or pathlib.Path
            The flat results table from ``BaseBenchmark.run()`` or a path to a
            result artifact (see ``evaluate``).

        Returns
        -------
        pandas.DataFrame
            Score matrix, index = ``validation_id``, columns = ``model_id``,
            values = the per-task mean of the selected metric.

        Raises
        ------
        ValueError
            If any ``(model_id, validation_id)`` cell is missing, i.e. an
            estimator has no score on some task (e.g. a failed experiment).
        """
        df = self._load_results(results)
        metric = self._resolve_metric(df)
        column = f"{metric}{_MEAN_SUFFIX}"
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in results. Available columns: "
                f"{list(df.columns)}."
            )
        scores = df.pivot(index="validation_id", columns="model_id", values=column)
        self._check_no_missing(scores)
        return scores

    @staticmethod
    def _check_no_missing(scores):
        """Raise if any (model, task) score is missing (NaN) in the matrix."""
        if not scores.isna().to_numpy().any():
            return
        missing = [
            (str(model), str(task))
            for task in scores.index
            for model in scores.columns
            if pd.isna(scores.at[task, model])
        ]
        raise ValueError(
            "Score matrix contains missing values; every estimator must have a "
            "score on every task before a post-hoc analysis can run. Missing "
            f"(model_id, validation_id) pairs: {missing}."
        )

    @staticmethod
    def _load_results(results):
        """Return a flat results DataFrame from a DataFrame or artifact path."""
        if isinstance(results, pd.DataFrame):
            return results

        # treat as a path to a storage-handler artifact; reconstruct the flat
        # ``ResultObject.to_dataframe`` schema by reusing the storage handlers.
        from sktime.benchmarking._storage_handlers import get_storage_backend

        handler_cls = get_storage_backend(results)
        result_objects = handler_cls(results).load()
        if len(result_objects) == 0:
            raise ValueError(f"No benchmark results found at '{results}'.")
        return pd.concat([r.to_dataframe() for r in result_objects], ignore_index=True)

    def _resolve_metric(self, df):
        """Resolve the metric name, inferring it if a single one is present."""
        if self.metric is not None:
            return self.metric

        candidates = []
        for col in df.columns:
            if col.endswith(_MEAN_SUFFIX):
                base = col[: -len(_MEAN_SUFFIX)]
                if base not in _NON_METRIC_BASES:
                    candidates.append(base)
        # preserve order, drop duplicates
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) == 0:
            raise ValueError(
                "No '<metric>_mean' columns found in results; cannot infer "
                "`metric`. Pass `metric` explicitly."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple metrics found in results: {candidates}. Pass `metric` "
                "explicitly to select one."
            )
        return candidates[0]

    @staticmethod
    def _as_estimator_dict(scores):
        """Map a score matrix to the legacy ``{model_id: [scores]}`` dict form."""
        return {col: scores[col].to_numpy() for col in scores.columns}

    def _mean_ranks(self, scores):
        """Average rank of each estimator across datasets.

        Ranks estimators within each dataset (best = rank 1 when
        ``lower_is_better``) and averages the ranks across datasets.

        Returns
        -------
        pandas.DataFrame
            Columns ``["model_id", "rank"]``, sorted by ascending rank.
        """
        ranks = scores.rank(axis=1, ascending=self.lower_is_better)
        mean_ranks = ranks.mean(axis=0).reset_index()
        mean_ranks.columns = ["model_id", "rank"]
        return mean_ranks.sort_values("rank").reset_index(drop=True)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the evaluator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        dict
            Parameters to construct a test instance.
        """
        return {}
