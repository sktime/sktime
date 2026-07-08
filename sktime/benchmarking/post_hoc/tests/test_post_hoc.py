"""Tests for the v2 post-hoc benchmark evaluators."""

__author__ = ["viktorkaz", "mloning"]

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.benchmarking.post_hoc import (
    CriticalDifferenceDiagram,
    FriedmanEvaluator,
    NemenyiEvaluator,
    RankEvaluator,
    RanksumEvaluator,
    SignTestEvaluator,
    TTestEvaluator,
    WilcoxonEvaluator,
)
from sktime.tests.test_switch import run_test_module_changed

# metric name used in the synthetic v2 results table
METRIC = "MeanSquaredError"

# 3 estimators x 5 datasets, distinct values (no ties -> no degenerate tests)
SCORES_BY_MODEL = {
    "model_A": [0.10, 0.20, 0.15, 0.18, 0.12],
    "model_B": [0.30, 0.25, 0.35, 0.28, 0.31],
    "model_C": [0.20, 0.22, 0.19, 0.25, 0.21],
}

ALL_EVALUATORS = [
    RankEvaluator,
    FriedmanEvaluator,
    NemenyiEvaluator,
    WilcoxonEvaluator,
    SignTestEvaluator,
    RanksumEvaluator,
    TTestEvaluator,
    CriticalDifferenceDiagram,
]


def _make_results_df(scores_by_model=SCORES_BY_MODEL, metric=METRIC):
    """Build a flat v2-style results DataFrame from a score matrix.

    Mirrors the schema of ``ResultObject.to_dataframe`` (one row per
    ``(model_id, validation_id)`` pair), including non-metric ``*_mean``
    timing columns that the evaluators must ignore.
    """
    n_datasets = len(next(iter(scores_by_model.values())))
    datasets = [f"task_{i}" for i in range(n_datasets)]
    rows = []
    for model, values in scores_by_model.items():
        for dataset, value in zip(datasets, values):
            rows.append(
                {
                    "validation_id": dataset,
                    "model_id": model,
                    f"{metric}_fold_0_test": value,
                    f"{metric}_mean": value,
                    f"{metric}_std": 0.0,
                    "fit_time_mean": 0.1,
                    "pred_time_mean": 0.1,
                    "runtime_secs": 0.2,
                }
            )
    return pd.DataFrame(rows)


pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)


# --------------------------------------------------------------------------- #
# input contract / adapter
# --------------------------------------------------------------------------- #
def test_metric_inference():
    """A single metric column is inferred when ``metric`` is not given."""
    df = _make_results_df()
    scores = FriedmanEvaluator()._coerce_to_score_matrix(df)
    assert list(scores.columns) == ["model_A", "model_B", "model_C"]
    assert scores.shape == (5, 3)


def test_metric_inference_ambiguous_raises():
    """Multiple metric columns require an explicit ``metric``."""
    df = _make_results_df()
    df["OtherMetric_mean"] = 1.0
    with pytest.raises(ValueError, match="Multiple metrics"):
        FriedmanEvaluator()._coerce_to_score_matrix(df)


def test_explicit_metric_selection():
    """An explicit ``metric`` selects the right column among several."""
    df = _make_results_df()
    df["OtherMetric_mean"] = 1.0
    scores = FriedmanEvaluator(metric=METRIC)._coerce_to_score_matrix(df)
    assert scores.shape == (5, 3)


def test_load_from_csv_path(tmp_path):
    """``evaluate`` accepts a CSV artifact path, not just a DataFrame."""
    from sktime.benchmarking._benchmarking_dataclasses import FoldResults, ResultObject
    from sktime.benchmarking._storage_handlers import CSVStorageHandler

    result_objects = []
    for model, values in SCORES_BY_MODEL.items():
        for i, value in enumerate(values):
            folds = {0: FoldResults(scores={METRIC: value})}
            result_objects.append(
                ResultObject(model_id=model, task_id=f"task_{i}", folds=folds)
            )
    path = tmp_path / "results.csv"
    CSVStorageHandler(path).save(result_objects)

    from_path = FriedmanEvaluator(metric=METRIC).evaluate(path)
    from_df = FriedmanEvaluator(metric=METRIC).evaluate(_make_results_df())
    pd.testing.assert_frame_equal(from_path, from_df)


def test_missing_scores_raise():
    """A missing (model, task) score is rejected during coercion."""
    df = _make_results_df()
    # drop model_C's score on task_0 -> a hole in the pivoted matrix
    df = df[~((df.model_id == "model_C") & (df.validation_id == "task_0"))]
    with pytest.raises(ValueError, match="missing values"):
        FriedmanEvaluator().evaluate(df)


# --------------------------------------------------------------------------- #
# parity with the underlying scipy computations (the legacy Evaluator math)
# --------------------------------------------------------------------------- #
def test_friedman_parity():
    """FriedmanEvaluator matches ``scipy.stats.friedmanchisquare``."""
    from scipy.stats import friedmanchisquare

    out = FriedmanEvaluator().evaluate(_make_results_df())
    stat, p = friedmanchisquare(*SCORES_BY_MODEL.values())
    assert out.loc[0, "statistic"] == pytest.approx(stat)
    assert out.loc[0, "p_value"] == pytest.approx(p)


def test_wilcoxon_parity():
    """WilcoxonEvaluator matches ``scipy.stats.wilcoxon`` per pair."""
    from scipy.stats import wilcoxon

    out = WilcoxonEvaluator().evaluate(_make_results_df())
    # 3 estimators -> 3 unique unordered pairs
    assert len(out) == 3
    row = out[(out.estimator_1 == "model_A") & (out.estimator_2 == "model_B")].iloc[0]
    w, p = wilcoxon(SCORES_BY_MODEL["model_A"], SCORES_BY_MODEL["model_B"])
    assert row["statistic"] == pytest.approx(w)
    assert row["p_val"] == pytest.approx(p)


def test_ranksum_parity():
    """RanksumEvaluator matches ``scipy.stats.ranksums`` per pair."""
    from scipy.stats import ranksums

    out = RanksumEvaluator().evaluate(_make_results_df())
    # ordered pairs incl. self -> 3 * 3 = 9 rows
    assert len(out) == 9
    row = out[(out.estimator_1 == "model_A") & (out.estimator_2 == "model_C")].iloc[0]
    s, p = ranksums(SCORES_BY_MODEL["model_A"], SCORES_BY_MODEL["model_C"])
    assert row["statistic"] == pytest.approx(s)
    assert row["p_val"] == pytest.approx(p)


def test_ttest_parity_and_bonferroni():
    """TTestEvaluator matches ``ttest_ind`` and adds a Bonferroni flag."""
    from scipy.stats import ttest_ind

    out = TTestEvaluator().evaluate(_make_results_df())
    assert "significant" not in out.columns
    row = out[(out.estimator_1 == "model_A") & (out.estimator_2 == "model_B")].iloc[0]
    s, p = ttest_ind(
        np.asarray(SCORES_BY_MODEL["model_A"]),
        np.asarray(SCORES_BY_MODEL["model_B"]),
    )
    assert row["statistic"] == pytest.approx(s)
    assert row["p_val"] == pytest.approx(p)

    corrected = TTestEvaluator(correction="bonferroni", alpha=0.05).evaluate(
        _make_results_df()
    )
    assert "significant" in corrected.columns
    threshold = 0.05 / (len(SCORES_BY_MODEL) ** 2)
    assert (corrected["significant"] == (corrected["p_val"] <= threshold)).all()


def test_sign_test_parity():
    """SignTestEvaluator matches a direct binomial test."""
    from scipy import stats

    binom = stats.binomtest if hasattr(stats, "binomtest") else stats.binom_test
    out = SignTestEvaluator().evaluate(_make_results_df())
    assert len(out) == 9  # ordered pairs incl. self
    a = np.asarray(SCORES_BY_MODEL["model_A"])
    b = np.asarray(SCORES_BY_MODEL["model_B"])
    expected = binom(int(np.sum(a > b)), len(a)).pvalue
    row = out[(out.estimator_1 == "model_A") & (out.estimator_2 == "model_B")].iloc[0]
    assert row["p_val"] == pytest.approx(expected)


def test_rank_evaluator():
    """RankEvaluator returns one mean rank per model, best first."""
    out = RankEvaluator().evaluate(_make_results_df())
    assert list(out.columns) == ["model_id", "rank"]
    assert len(out) == 3
    # model_A has the lowest errors on every dataset -> best (rank 1) when
    # lower_is_better=True
    assert out.iloc[0]["model_id"] == "model_A"
    assert out.iloc[0]["rank"] == pytest.approx(1.0)
    # average rank over 3 estimators is 2.0
    assert out["rank"].mean() == pytest.approx(2.0)


@pytest.mark.skipif(
    not _check_soft_dependencies("scikit_posthocs", severity="none"),
    reason="scikit_posthocs not installed",
)
def test_nemenyi_evaluator():
    """NemenyiEvaluator returns a square pairwise p-value matrix."""
    out = NemenyiEvaluator().evaluate(_make_results_df())
    assert out.shape == (3, 3)


def test_nemenyi_missing_dependency_raises():
    """NemenyiEvaluator raises a clean error if scikit_posthocs is missing."""
    if _check_soft_dependencies("scikit_posthocs", severity="none"):
        pytest.skip("scikit_posthocs is installed")
    with pytest.raises(ModuleNotFoundError):
        NemenyiEvaluator().evaluate(_make_results_df())


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="matplotlib not installed",
)
def test_critical_difference_diagram():
    """CriticalDifferenceDiagram returns ranks and a (fig, ax) plot."""
    import matplotlib

    matplotlib.use("Agg")

    cd = CriticalDifferenceDiagram()
    ranks = cd.evaluate(_make_results_df())
    assert list(ranks.columns) == ["model_id", "rank"]
    assert len(ranks) == 3

    fig, ax = cd.plot(_make_results_df())
    assert fig is not None and ax is not None


# --------------------------------------------------------------------------- #
# generic construction / statelessness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("evaluator_cls", ALL_EVALUATORS)
def test_construct_via_test_params(evaluator_cls):
    """Every evaluator is constructible from ``get_test_params``."""
    params = evaluator_cls.get_test_params()
    if isinstance(params, list):
        params = params[0]
    evaluator = evaluator_cls(**params)
    assert evaluator.get_params() is not None


def test_evaluator_is_stateless():
    """Repeated calls do not mutate the evaluator and give identical results."""
    evaluator = FriedmanEvaluator()
    df = _make_results_df()
    first = evaluator.evaluate(df)
    second = evaluator.evaluate(df)
    pd.testing.assert_frame_equal(first, second)
