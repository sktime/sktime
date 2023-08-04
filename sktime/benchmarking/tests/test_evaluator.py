"""Test evaluator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric
from sktime.benchmarking.results import RAMResults
from sktime.series_as_features.model_selection import PresplitFilesCV
from sktime.utils.validation._dependencies import _check_soft_dependencies


def dummy_results():
    """Results that are dummy."""
    results = RAMResults()
    results.cv = PresplitFilesCV()
    results.save_predictions(
        strategy_name="alg1",
        dataset_name="dataset1",
        index=np.array([1, 2, 3, 4]),
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 1, 1]),
        y_proba=None,
        cv_fold=0,
        train_or_test="test",
        fit_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        fit_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
        predict_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        predict_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
    )
    results.save_predictions(
        strategy_name="alg1",
        dataset_name="dataset2",
        index=np.array([1, 2, 3, 4]),
        y_true=np.array([0, 0, 0, 0]),
        y_pred=np.array([0, 0, 0, 0]),
        y_proba=None,
        cv_fold=0,
        train_or_test="test",
        fit_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        fit_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
        predict_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        predict_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
    )

    results.save_predictions(
        strategy_name="alg2",
        dataset_name="dataset1",
        index=np.array([1, 2, 3, 4]),
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([0, 0, 0, 0]),
        y_proba=None,
        cv_fold=0,
        train_or_test="test",
        fit_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        fit_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
        predict_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        predict_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
    )
    results.save_predictions(
        strategy_name="alg2",
        dataset_name="dataset2",
        index=np.array([1, 2, 3, 4]),
        y_true=np.array([0, 0, 0, 0]),
        y_pred=np.array([1, 1, 1, 1]),
        y_proba=None,
        cv_fold=0,
        train_or_test="test",
        fit_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        fit_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
        predict_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        predict_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
    )

    results.save_predictions(
        strategy_name="alg3",
        dataset_name="dataset1",
        index=np.array([1, 2, 3, 4]),
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 0, 1]),
        y_proba=None,
        cv_fold=0,
        train_or_test="test",
        fit_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        fit_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
        predict_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        predict_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
    )
    results.save_predictions(
        strategy_name="alg3",
        dataset_name="dataset2",
        index=np.array([1, 2, 3, 4]),
        y_true=np.array([0, 0, 0, 0]),
        y_pred=np.array([0, 0, 1, 0]),
        y_proba=None,
        cv_fold=0,
        train_or_test="test",
        fit_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        fit_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
        predict_estimator_start_time=pd.to_datetime(1605268800, unit="ms"),
        predict_estimator_end_time=pd.to_datetime(1605268801, unit="ms"),
    )
    return results


def evaluator_setup(score_function):
    """Set up the evaluator."""
    evaluator = Evaluator(dummy_results())
    metric = PairwiseMetric(func=score_function, name="score_function")
    metrics_by_strategy = evaluator.evaluate(metric=metric)

    return evaluator, metrics_by_strategy


def test_rank():
    """Test rank."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    expected_ranks = pd.DataFrame(
        {
            "strategy": ["alg1", "alg2", "alg3"],
            "score_function_mean_rank": [1.0, 3.0, 2.0],
        }
    )
    generated_ranks = evaluator.rank()
    assert expected_ranks.equals(generated_ranks)


def test_accuracy_score():
    """Test accuracy score."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)

    expected_accuracy = pd.DataFrame(
        {
            "strategy": ["alg1", "alg2", "alg3"],
            "score_function_mean": [1.00, 0.00, 0.75],
            "score_function_stderr": [0.00, 0.00, 0.25],
        }
    )

    assert metrics_by_strategy.equals(expected_accuracy)


def test_sign_test():
    """Test sign_test."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    results = evaluator.sign_test()[1].values
    expected = np.full((3, 3), 0.5)
    assert np.array_equal(expected, results)


def test_ranksum_test():
    """Test ranksum_test."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    expected = np.array(
        [
            [0.0, 1.0, 1.54919334, 0.12133525, 1.54919334, 0.12133525],
            [-1.54919334, 0.12133525, 0.0, 1.0, -1.54919334, 0.12133525],
            [-1.54919334, 0.12133525, 1.54919334, 0.12133525, 0.0, 1.0],
        ]
    )
    expected = np.round(expected, 3)
    results = np.round(evaluator.ranksum_test()[1].values, 3)
    assert np.array_equal(results, expected)


def test_t_test_bonfer():
    """Test t_test_bonfer."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    expected = np.array([[False, True, True], [True, False, True], [True, True, False]])
    result = evaluator.t_test_with_bonferroni_correction().values
    assert np.array_equal(expected, result)


@pytest.mark.skipif(
    not _check_soft_dependencies("scikit_posthocs", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_nemenyi():
    """Test nemenyi."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    expected = np.array(
        [
            [1.0, 0.082085, 0.53526143],
            [0.082085, 1.0, 0.53526143],
            [0.53526143, 0.53526143, 1.0],
        ]
    )
    expected = np.round(expected, 3)
    result = np.round(evaluator.nemenyi().values, 3)

    return np.array_equal(expected, result)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.xfail(reason="known sporadic failure of unknown cause, see #2368")
def test_plots():
    """Test plots."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    evaluator.plot_boxplots()
    evaluator.t_test()


def test_wilcoxon():
    """Test wilcoxon."""
    expected = pd.DataFrame(
        data={"statistic": [0.0, 0.0, 0.0], "p_val": [0.5, 0.5, 0.5]}
    )
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    results = evaluator.wilcoxon_test()
    assert results[expected.columns].equals(expected)


def test_run_times():
    """Test run_times."""
    evaluator, metrics_by_strategy = evaluator_setup(score_function=accuracy_score)
    result = evaluator.fit_runtime()
    expected = np.array(
        [
            [0.001, 0.001],
            [0.001, 0.001],
            [0.001, 0.001],
            [0.001, 0.001],
            [0.001, 0.001],
            [0.001, 0.001],
        ]
    )
    assert np.array_equal(expected, result)
