# -*- coding: utf-8 -*-
"""Classifying benchmarks tests."""
import pandas as pd
from sklearn.model_selection import KFold

from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_arrow_head


def test_classificationbenchmark(tmp_path):
    """Test benchmarking a classifier estimator."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(
        estimator=RocketClassifier(),
        estimator_id="RocketClassifier-v1",
    )

    # Specify cross-validation split methods
    cv_splitter = KFold(n_splits=2, shuffle=False)

    # Specify comparison metrics
    scorers = ["accuracy"]

    # Specify dataset loaders
    dataset_loaders = [load_arrow_head]

    # Add tasks, optionally use loops etc. to easily set up multiple tasks
    for dataset_loader in dataset_loaders:
        benchmark.add_task(
            dataset_loader,
            cv_splitter,
            scorers,
        )
    result_file = tmp_path / "classification_results.csv"
    results_df = benchmark.run(result_file)
    results_df = pd.read_csv(result_file)
    results_df = results_df.drop(columns=["runtime_secs"])

    expected_result = pd.DataFrame(
        {
            "validation_id": {0: "[dataset=load_arrow_head]_[cv_splitter=KFold]-v1"},
            "model_id": {0: "RocketClassifier-v1"},
            "accuracy_fold_0_test": {0: 0.2264150943396226},
            "accuracy_fold_1_test": {0: 0.9047619047619048},
            "accuracy_mean": {0: 0.5655884995507637},
            "accuracy_std": {0: 0.479663629645861},
        }
    )

    pd.testing.assert_frame_equal(
        expected_result, results_df, check_exact=False, atol=0, rtol=1e-3
    )
    return None
