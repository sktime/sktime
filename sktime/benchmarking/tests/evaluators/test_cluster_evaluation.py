# -*- coding: utf-8 -*-
"""Test cluster evaluation."""
# -*- coding: utf-8 -*-
import os.path
import shutil

from sktime.benchmarking.evaluators import ClusterEvaluator
from sktime.datasets import load_acsf1, load_arrow_head, load_osuleaf
from sktime.datatypes import convert_to


def _test_cluster_evaluation():

    datasets = [
        ("acsf1", load_acsf1),
        ("arrowhead", load_arrow_head),
        ("osuleaf", load_osuleaf),
    ]

    dataset_paths = {}

    for dataset in datasets:
        dataset_name = dataset[0]
        load_dataset = dataset[1]
        X_train, y_train = load_dataset(split="train")
        X_test, y_test = load_dataset(split="test")

        num_data_train = len(X_train)
        num_data_test = len(X_test)

        max_num_data_points = 20

        if num_data_train > max_num_data_points:
            num_data_train = max_num_data_points

        if num_data_test > max_num_data_points:
            num_data_test = max_num_data_points

        trainX = convert_to(X_train[0:num_data_train], "numpy3D")
        testX = convert_to(X_test[0:num_data_test], "numpy3D")
        dataset_paths[dataset_name] = [trainX, testX]

    evaluator = ClusterEvaluator(
        "test_results",
        os.path.abspath("./result_out"),
        experiment_name="example_experiment",
        naming_parameter_key="metric",
        critical_diff_params={"alpha": 100000.0},
        metrics="all",
        dataset_paths=dataset_paths,
    )

    evaluator.run_evaluation(["kmeans", "kmedoids"])

    shutil.rmtree(os.path.abspath("./result_out"))
