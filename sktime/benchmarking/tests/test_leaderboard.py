"""Tests for the benchmark leaderboard."""

import warnings

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sktime.benchmarking._leaderboard import _build_leaderboard
from sktime.benchmarking.analysis import AverageRank
from sktime.benchmarking.benchmarks import BenchmarkingResults
from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.classification.dummy import DummyClassifier
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    make_forecasting_scorer,
)
from sktime.split import ExpandingWindowSplitter
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.panel import make_classification_problem

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)

# Two tasks, three models. MAE is lower-is-better, ACC is higher-is-better.
# MAE ranks: task_0 A=1, B=2, C=3; task_1 B=1, C=2, A=3.
# ACC ranks: task_0 C=1, A=2, B=3; task_1 B=1, A=2, C=3.
_ROWS = [
    # validation_id, model_id, MAE, ACC, fit, pred
    ("task_0", "A", 0.1, 0.8, 1.0, 0.1),
    ("task_0", "B", 0.2, 0.7, 2.0, 0.4),
    ("task_0", "C", 0.3, 0.9, 4.0, 0.5),
    ("task_1", "A", 0.4, 0.6, 5.0, 0.3),
    ("task_1", "B", 0.2, 0.7, 2.0, 0.4),
    ("task_1", "C", 0.3, 0.5, 6.0, 0.1),
]


def _results_frame(rows=_ROWS):
    """Flat v2 results table with one row per task and model."""
    records = []
    for validation_id, model_id, mae, acc, fit_time, pred_time in rows:
        records.append(
            {
                "validation_id": validation_id,
                "model_id": model_id,
                "MAE_mean": mae,
                "ACC_mean": acc,
                "fit_time_mean": fit_time,
                "pred_time_mean": pred_time,
            }
        )
    return pd.DataFrame(records)


def _board(**kwargs):
    """Leaderboard with explicit directions, so inference is not involved."""
    kwargs.setdefault("lower_is_better", {"MAE": True, "ACC": False})
    return _build_leaderboard(_results_frame(), **kwargs)


def test_each_scorer_keeps_its_own_rank():
    """Several scorers report a rank each and are not averaged together."""
    board = _board()

    # MAE ranks across tasks: A=2, B=1.5, C=2.5, so rows follow MAE.
    # ACC ranks across tasks: A=2, B=2, C=2. Those are not mixed into MAE.
    assert "mean_rank" not in board.columns
    assert "rank" not in board.columns
    assert board["model_id"].tolist() == ["B", "A", "C"]
    assert board["MAE_rank"].tolist() == pytest.approx([1.5, 2.0, 2.5])
    assert board["ACC_rank"].tolist() == pytest.approx([2.0, 2.0, 2.0])

    assert list(board.columns) == [
        "model_id",
        "MAE_rank",
        "ACC_rank",
        "MAE",
        "ACC",
        "fit_time_s",
        "predict_time_s",
    ]
    row_b = board.loc[board["model_id"] == "B"].iloc[0]
    assert row_b["MAE"] == pytest.approx(0.2)
    assert row_b["ACC"] == pytest.approx(0.7)
    assert row_b["fit_time_s"] == pytest.approx(2.0)
    assert row_b["predict_time_s"] == pytest.approx(0.4)
    assert row_b["MAE_rank"] == pytest.approx(1.5)
    assert row_b["ACC_rank"] == pytest.approx(2.0)


def test_single_scorer_matches_average_rank_analyzer():
    """One scorer's ``<scorer>_rank`` reproduces AverageRank."""
    frame = _results_frame()
    board = _build_leaderboard(frame, scorers="MAE", lower_is_better=True)
    expected = AverageRank(metric="MAE", lower_is_better=True).evaluate(frame)

    got = board.set_index("model_id")["MAE_rank"].sort_index()
    ref = expected.set_index("model_id")["rank"].sort_index()
    pd.testing.assert_series_equal(got, ref, check_names=False)
    assert "mean_rank" not in board.columns
    assert "ACC" not in board.columns


def test_single_dataset_orders_by_score():
    """One task is ordered by that task's score, not by the cross-task mean."""
    board = _board(dataset="task_0", scorers="MAE", lower_is_better=True)

    assert board["model_id"].tolist() == ["A", "B", "C"]
    assert board["MAE_rank"].tolist() == pytest.approx([1, 2, 3])
    assert list(board.columns) == [
        "model_id",
        "MAE_rank",
        "MAE",
        "fit_time_s",
        "predict_time_s",
    ]
    assert board["MAE"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert board["fit_time_s"].tolist() == pytest.approx([1.0, 2.0, 4.0])
    assert board["predict_time_s"].tolist() == pytest.approx([0.1, 0.4, 0.5])


def test_dataset_name_and_task_id_select_the_same_tasks():
    """A dataset name and a full task id both select the intended tasks."""
    rows = [
        (
            "[dataset=Longley]_[cv_splitter=ExpandingWindowSplitter]",
            "A",
            0.2,
            0.5,
            1.0,
            0.1,
        ),
        (
            "[dataset=Longley]_[cv_splitter=SlidingWindowSplitter]",
            "A",
            0.4,
            0.5,
            3.0,
            0.3,
        ),
        (
            "[dataset=Airline]_[cv_splitter=ExpandingWindowSplitter]",
            "A",
            1.0,
            0.5,
            9.0,
            0.9,
        ),
        (
            "[dataset=Longley]_[cv_splitter=ExpandingWindowSplitter]",
            "B",
            0.1,
            0.5,
            1.0,
            0.1,
        ),
        (
            "[dataset=Longley]_[cv_splitter=SlidingWindowSplitter]",
            "B",
            0.1,
            0.5,
            1.0,
            0.1,
        ),
        (
            "[dataset=Airline]_[cv_splitter=ExpandingWindowSplitter]",
            "B",
            0.5,
            0.5,
            1.0,
            0.1,
        ),
    ]
    frame = _results_frame(rows)
    by_name = _build_leaderboard(frame, dataset="Longley", lower_is_better=True)
    by_ids = _build_leaderboard(
        frame,
        dataset=[
            "[dataset=Longley]_[cv_splitter=ExpandingWindowSplitter]",
            "[dataset=Longley]_[cv_splitter=SlidingWindowSplitter]",
        ],
        lower_is_better=True,
    )
    pd.testing.assert_frame_equal(by_name, by_ids)
    # Longley MAE means: A=(0.2+0.4)/2=0.3, B=(0.1+0.1)/2=0.1, so B ranks first.
    assert by_name["model_id"].tolist() == ["B", "A"]
    one_task = _build_leaderboard(
        frame,
        dataset="[dataset=Airline]_[cv_splitter=ExpandingWindowSplitter]",
        scorers="MAE",
        lower_is_better=True,
    )
    assert one_task["MAE"].tolist() == pytest.approx([0.5, 1.0])


def test_multiple_scorers_on_one_dataset_keep_separate_ranks():
    """On one task, each scorer's rank follows that scorer's score."""
    board = _board(dataset="task_0")
    # MAE ranks A=1, B=2, C=3, and that order is the row order.
    # ACC ranks C=1, A=2, B=3.
    assert "mean_rank" not in board.columns
    assert board["model_id"].tolist() == ["A", "B", "C"]
    assert board["MAE_rank"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert board["ACC_rank"].tolist() == pytest.approx([2.0, 3.0, 1.0])


def test_tied_scores_share_a_rank():
    """Tied scores share the average of the tied positions."""
    rows = [
        ("task_0", "A", 0.1, 0.0, 1.0, 0.1),
        ("task_0", "B", 0.1, 0.0, 1.0, 0.1),
        ("task_0", "C", 0.2, 0.0, 1.0, 0.1),
    ]
    board = _build_leaderboard(
        _results_frame(rows), scorers="MAE", lower_is_better=True
    )
    assert board["model_id"].tolist() == ["A", "B", "C"]
    assert board["MAE_rank"].tolist() == pytest.approx([1.5, 1.5, 3.0])
    assert "mean_rank" not in board.columns


def test_lower_is_better_override_reverses_order():
    """An explicit direction replaces the inferred one for every scorer."""
    higher = _build_leaderboard(
        _results_frame(), dataset="task_0", scorers="ACC", lower_is_better=False
    )
    forced_lower = _build_leaderboard(
        _results_frame(), dataset="task_0", scorers="ACC", lower_is_better=True
    )
    assert higher["model_id"].tolist() == ["C", "A", "B"]
    assert forced_lower["model_id"].tolist() == ["B", "A", "C"]


def test_unknown_dataset_reports_available_ids():
    """An unmatched dataset name raises and names the tasks that do exist."""
    with pytest.raises(ValueError, match="No results matched dataset"):
        _board(dataset="missing")


def test_missing_scores_raise():
    """A model without a score on a selected task is rejected."""
    frame = _results_frame().drop(index=[5])  # C on task_1
    with pytest.raises(ValueError, match="missing scores"):
        _build_leaderboard(frame, scorers="MAE", lower_is_better=True)


def test_unknown_scorer_name_warns_and_assumes_lower_is_better():
    """A scorer with no known direction warns and treats lower scores as better."""
    frame = _results_frame().rename(columns={"MAE_mean": "custom_gain_mean"})
    frame = frame.drop(columns=["ACC_mean"])
    with pytest.warns(UserWarning, match="custom_gain"):
        board = _build_leaderboard(frame, dataset="task_0")
    # Values 0.1, 0.2, 0.3: lower-is-better puts A first.
    assert board["model_id"].tolist() == ["A", "B", "C"]


def test_registered_metric_direction_is_inferred():
    """sktime metric classes supply their own direction, without a warning."""
    lower = pd.DataFrame(
        {
            "validation_id": ["task_0", "task_0"],
            "model_id": ["good", "bad"],
            "MeanAbsoluteError_mean": [0.1, 0.5],
        }
    )
    higher = pd.DataFrame(
        {
            "validation_id": ["task_0", "task_0"],
            "model_id": ["good", "bad"],
            "EmpiricalCoverage_mean": [0.9, 0.2],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        lower_board = _build_leaderboard(lower)
        higher_board = _build_leaderboard(higher)
    assert lower_board["model_id"].tolist() == ["good", "bad"]
    assert higher_board["model_id"].tolist() == ["good", "bad"]


def test_accuracy_score_direction_from_name():
    """sklearn ``accuracy_score`` ranks a higher score as better."""
    frame = pd.DataFrame(
        {
            "validation_id": ["task_0", "task_0"],
            "model_id": ["good", "bad"],
            "accuracy_score_mean": [0.9, 0.4],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        board = _build_leaderboard(frame)
    assert board["model_id"].tolist() == ["good", "bad"]


def test_leaderboard_before_run_raises():
    """Calling leaderboard without results raises a clear error."""
    benchmark = ClassificationBenchmark()
    with pytest.raises(ValueError, match="Call `run` first"):
        benchmark.leaderboard()


def test_benchmark_uses_registered_scorer_direction():
    """A custom metric's tag decides the direction when results are passed in."""

    def _gain(y_true, y_pred):
        return 0.0

    metric = make_forecasting_scorer(
        func=_gain, name="custom_gain", greater_is_better=True
    )
    benchmark = ForecastingBenchmark()
    benchmark.add_task(
        lambda: pd.Series([1.0, 2.0, 3.0, 4.0]),
        ExpandingWindowSplitter(initial_window=2, fh=1),
        [metric],
    )
    frame = pd.DataFrame(
        {
            "validation_id": ["task_0", "task_0"],
            "model_id": ["high", "low"],
            "custom_gain_mean": [5.0, 1.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        board = benchmark.leaderboard(results=frame)
    assert board["model_id"].tolist() == ["high", "low"]


def test_leaderboard_after_run_and_from_saved_file(tmp_path):
    """A completed run and its saved file produce the same leaderboard."""

    def loader_trend():
        return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

    def loader_flat():
        return pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float)

    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(NaiveForecaster(strategy="last"), "last")
    benchmark.add_estimator(NaiveForecaster(strategy="mean"), "mean")
    cv = ExpandingWindowSplitter(initial_window=3, step_length=1, fh=1)
    scorers = [MeanAbsoluteError(), MeanAbsolutePercentageError()]
    benchmark.add_task(loader_trend, cv, scorers)
    benchmark.add_task(loader_flat, cv, scorers)

    path = tmp_path / "results.csv"
    results = benchmark.run(path)
    board = benchmark.leaderboard()
    from_file = BenchmarkingResults(path).leaderboard()
    from_path_arg = benchmark.leaderboard(results=path)

    pd.testing.assert_frame_equal(board, from_file)
    pd.testing.assert_frame_equal(board, from_path_arg)
    assert set(board["model_id"]) == {"last", "mean"}
    assert "mean_rank" not in board.columns
    assert board["MeanAbsoluteError_rank"].is_monotonic_increasing
    assert {
        "MeanAbsoluteError",
        "MeanAbsoluteError_rank",
        "MeanAbsolutePercentageError",
        "MeanAbsolutePercentageError_rank",
    } <= set(board.columns)

    one = benchmark.leaderboard(dataset="loader_trend", scorers=MeanAbsoluteError())
    subset = results[
        results["validation_id"].str.contains("loader_trend", regex=False)
    ].sort_values("MeanAbsoluteError_mean")
    assert one["model_id"].tolist() == subset["model_id"].tolist()
    assert len(one) == 2


def test_classification_leaderboard_prefers_higher_accuracy():
    """Accuracy from a classification run ranks the higher score first."""

    def loader():
        X, y = make_classification_problem(
            n_instances=12, n_timepoints=6, n_classes=2, random_state=0
        )
        y[:] = 0
        y.iloc[:2] = 1
        return X, y

    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(DummyClassifier(strategy="most_frequent"), "frequent")
    benchmark.add_estimator(
        DummyClassifier(strategy="constant", constant=1), "constant"
    )
    benchmark.add_task(loader, KFold(n_splits=2), [accuracy_score])
    benchmark.run()
    board = benchmark.leaderboard()

    assert board["model_id"].iloc[0] == "frequent"
    assert board["accuracy_score"].iloc[0] > board["accuracy_score"].iloc[1]
    assert "fit_time_s" in board.columns
    assert "predict_time_s" in board.columns
