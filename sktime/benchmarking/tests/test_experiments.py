# -*- coding: utf-8 -*-
"""Functions to test the functions in experiments.py."""

import os.path

from sktime.benchmarking.experiments import run_classification_experiment
from sktime.benchmarking.experiments import run_clustering_experiment
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.clustering import TimeSeriesKMeans
from sktime.datasets import load_unit_test


def test_run_clustering_experiment():
    """Test running and saving results for clustering.

    Currently it just checks the files have been created, then deletes them.
    """
    dataset = "UnitTest"
    train_X, train_Y = load_unit_test("TRAIN", return_X_y=True)
    test_X, test_Y = load_unit_test("TEST", return_X_y=True)
    run_clustering_experiment(
        train_X,
        TimeSeriesKMeans(n_clusters=2),
        results_path="../Temp/",
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name="kmeans",
        dataset_name=dataset,
        resample_id=0,
    )
    test_path = f"../Temp/kmeans/Predictions/{dataset}/testResample0.csv"
    train_path = f"../Temp/kmeans/Predictions/{dataset}/trainResample0.csv"
    assert os.path.isfile(test_path)
    assert os.path.isfile(train_path)
    os.remove(test_path)
    os.remove(train_path)


def test_run_classification_experiment():
    """Test running and saving results for classifiers.

    Currently it just checks the files have been created, then deletes them.
    """
    dataset = "UnitTest"
    train_X, train_Y = load_unit_test("TRAIN", return_X_y=True)
    test_X, test_Y = load_unit_test("TEST", return_X_y=True)
    run_classification_experiment(
        train_X,
        train_Y,
        test_X,
        test_Y,
        TimeSeriesForestClassifier(n_estimators=10),
        "../Temp/",
        cls_name="TSF",
        dataset="UnitTest",
        resample_id=0,
        train_file=True,
    )
    test_path = f"../Temp/TSF/Predictions/{dataset}/testResample0.csv"
    train_path = f"../Temp/TSF/Predictions/{dataset}/trainResample0.csv"
    assert os.path.isfile(test_path)
    assert os.path.isfile(train_path)
    os.remove(test_path)
    os.remove(train_path)


# def test_load_and_run_clustering_experiment():
#     """Test loading, running and saving.
#
#     Currently it just checks that the files have been created, then deletes them.
#     Later it can be enhanced to check the results can be loaded.
#     """
#     load_and_run_clustering_experiment(
#         overwrite=True,
#         problem_path="../../datasets/data/",
#         results_path="../Temp/",
#         cls_name="kmeans",
#         dataset="UnitTest",
#         resample_id=0,
#         train_file=True,
#     )
#     assert os.path.isfile("../Temp/kmeans/Predictions/UnitTest/testResample0.csv")
#     assert os.path.isfile("../Temp/kmeans/Predictions/UnitTest/trainResample0.csv")
#     os.remove("../Temp/kmeans/Predictions/UnitTest/testResample0.csv")
#     os.remove("../Temp/kmeans/Predictions/UnitTest/trainResample0.csv")
#
