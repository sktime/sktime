# -*- coding: utf-8 -*-
"""Cluster Experiments.

code to run experiments for clustering, saving results in a standard format.
The main method is run_clustering_experiment. However, this file is also configured for
runs of the main method with command line arguments, or for single debugging runs.
"""
__author__ = ["TonyBagnall"]
import os
import sys

import sktime.datasets.tsc_dataset_names as dataset_lists
from sktime.benchmarking.experiments import (
    load_and_run_clustering_experiment,
    run_clustering_experiment,
)
from sktime.clustering import TimeSeriesKMeans
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts

# We sometimes want to force execution in a single thread. sklearn often threads in ways
# beyond the users control. This forces single thread execution, which is required,
# for example, when running on an HPC
# MUST be done before numpy import
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def demo_loading():
    """Test function to check dataset loading of univariate and multivaria problems."""
    for i in range(0, len(dataset_lists.univariate)):
        data_dir = "E:/tsc_ts/"
        dataset = dataset_lists.univariate[i]
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        print("Loaded " + dataset + " in position " + str(i))
        print("Train X shape :")
        print(trainX.shape)
        print("Train Y shape :")
        print(trainY.shape)
        print("Test X shape :")
        print(testX.shape)
        print("Test Y shape :")
        print(testY.shape)
    for i in range(16, len(dataset_lists.multivariate)):
        data_dir = "E:/mtsc_ts/"
        dataset = dataset_lists.multivariate[i]
        print("Loading " + dataset + " in position " + str(i) + ".......")
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        print("Loaded " + dataset)
        print("Train X shape :")
        print(trainX.shape)
        print("Train Y shape :")
        print(trainY.shape)
        print("Test X shape :")
        print(testX.shape)
        print("Test Y shape :")
        print(testY.shape)


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """
    if sys.argv.__len__() > 1:  # cluster run, this is fragile
        print(sys.argv)
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        classifier = sys.argv[3]
        dataset = sys.argv[4]
        resample = int(sys.argv[5]) - 1
        tf = str(sys.argv[6]) == "True"
        load_and_run_clustering_experiment(
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=classifier,
            dataset=dataset,
            resample_id=resample,
            train_file=tf,
        )
    else:  # Local run
        print(" Local Run")
        data_dir = "../datasets/data/"
        results_dir = "C:/Temp/Clusterers/"
        dataset = "UnitTest"
        clusterer = "kmeans"
        resample = 0
        tf = True
        clst = TimeSeriesKMeans(n_clusters=2)
        load_and_run_clustering_experiment(
            overwrite=True,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=clusterer,
            dataset=dataset,
            resample_id=resample,
            train_file=tf,
            clusterer=clst,
        )
        train_X, train_Y = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        test_X, test_Y = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        run_clustering_experiment(
            train_X,
            clst,
            results_path=results_dir + "Temp/",
            trainY=train_Y,
            testX=test_X,
            testY=test_Y,
            cls_name=clusterer,
        )
