# -*- coding: utf-8 -*-
"""Classifying benchmarks tests."""
import pandas as pd
from sklearn.model_selection import ShuffleSplit

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
    cv_splitter = ShuffleSplit(n_splits=2, test_size=0.3, random_state=42)

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

    expected = [
        0.9375,
        0.96875,
        0.875,
        0.953125,
        0.921875,
        0.984375,
        0.90625,
        1.00,
        0.890625,
        0.859375,
        0.84375,
    ]
    expected = pd.DataFrame({"accuracy": expected})

    assert results_df["accuracy_fold_0_test"].values in expected[["accuracy"]].values
    assert results_df["accuracy_fold_1_test"].values in expected[["accuracy"]].values
    return None
