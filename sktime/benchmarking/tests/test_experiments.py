# -*- coding: utf-8 -*-
"""Functions to test the functions in experiments.py."""

import os.path

from sktime.benchmarking.experiments import (
    load_and_run_clustering_experiment,
    run_clustering_experiment,
)
from sktime.clustering import TimeSeriesKMeans



def test_load_and_run_clustering_experiment():
    """Test loading, running and saving.
    Currently it just checks the files have been created, then deletes them. Later it
    can be enhanced to check the results can be loaded.
    """
    data_dir = "../datasets/data/"
    results_dir = "../Temp/"
    dataset = "UnitTest"
    clusterer = "kmeans"
    resample = 0
    tf = True
    load_and_run_clustering_experiment(
        overwrite=True,
        problem_path=data_dir,
        results_path=results_dir,
        cls_name=clusterer,
        dataset=dataset,
        resampleID=resample,
        train_file=tf,
        clusterer=clst,
    )
    assert os.path.isfile("../Temp/kmeans/Predictions/UnitTest/testResample0.csv")
    assert os.path.isfile("../Temp/kmeans/Predictions/UnitTest/trainResample0.csv")
    os.remove("../Temp/kmeans/Predictions/UnitTest/testResample0.csv")
    os.remove("../Temp/kmeans/Predictions/UnitTest/trainResample0.csv")


def test_run_clustering_experiment():
    data_dir = "../datasets/data/"
    results_dir = "../Temp/"
    dataset = "UnitTest"
    clusterer = "kmeans"
    resample = 0
    tf = True
    clst = TimeSeriesKMeans(n_clusters=2)
    train_X, train_Y = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
    test_X, test_Y = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
    run_clustering_experiment(
        train_X,
        clst,
        results_path=results_dir,
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name="kmeans2",
        dataset_name=dataset,
    )
    assert os.path.isfile("../Temp/kmeans2/Predictions/UnitTest/testResample0.csv")
    assert os.path.isfile("../Temp/kmeans2/Predictions/UnitTest/trainResample0.csv")
    os.remove("../Temp/kmeans2/Predictions/UnitTest/testResample0.csv")
    os.remove("../Temp/kmeans2/Predictions/UnitTest/trainResample0.csv")
