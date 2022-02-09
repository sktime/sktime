# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os
import sys

from sktime.contrib.set_classifier import set_classifier

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sktime.datasets.tsc_dataset_names as dataset_lists
from sktime.datasets import load_from_tsfile_to_dataframe as load_ts
from sktime.clustering import TimeSeriesKMeans, TimeSeriesKMedoids
from sktime.benchmarking.experiments import run_clustering_experiment

"""Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but isfrom sktime.classification.interval_based import (
    CanonicalIntervalForest,
 not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java.
"""


def demo_loading():
    """Test function to check dataset loading of univariate and multivaria problems."""
    for i in range(0, len(dataset_lists.univariate)):
        data_dir = "../"
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


def config_clusterer(clusterer, config, n_clusters, rand):
    """Configure the custerer for experiments."""
    if clusterer == "kmeans":
        if config != "":
            cls = TimeSeriesKMeans(n_clusters=n_clusters, metric=distance,
                                   init_algorithm="kmeans++",
                                   random_state=rand)
        else:
            cls = TimeSeriesKMeans(n_clusters=n_clusters, random_state=rand)
    elif clusterer == "kmedoids":
        if config != "":
            cls = TimeSeriesKMedoids(n_clusters=n_clusters, metric=distance,
                                     random_state=rand)
        else:
            cls = TimeSeriesKMedoids(n_clusters=n_clusters, random_state=rand)
    return cls


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    if sys.argv.__len__() > 1:  # cluster run, this is fragile
        print(sys.argv)
        data_dir = "/home/ajb/data/Univariate_ts/"
        results_dir = "/home/ajb/results/"
        clusterer = "kmeans"
        dataset = sys.argv[1]
        resample = int(sys.argv[2]) - 1
        tf = True
        distance = sys.argv[3]
        train_X, train_Y = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        test_X, test_Y = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        clst = config_clusterer(
            clusterer=clusterer, config=distance, n_clusters=len(set(train_Y)),
                                                                 rand=resample+1)
        run_clustering_experiment(
            train_X,
            clst,
            results_path=results_dir,
            trainY=train_Y,
            testX=test_X,
            testY=test_Y,
            cls_name=clusterer+"_"+distance,
            dataset_name=dataset,
            resample_id=resample,
        )
    else:  # Local run
        print(" Local Run")
        data_dir = "../datasets/data/"
        results_dir = "./temp"
        dataset = "GunPoint"
        clusterer = "kmeans"
        resample = 2
        tf = True
        distance = "euclidean"
        train_X, train_Y = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        test_X, test_Y = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        clst = config_clusterer(
            clusterer=clusterer, config=distance, n_clusters=len(set(train_Y)),
                                                                 rand=resample+1)
        run_clustering_experiment(
            train_X,
            clst,
            results_path=results_dir,
            trainY=train_Y,
            testX=test_X,
            testY=test_Y,
            cls_name=clusterer+"_"+distance,
            dataset_name=dataset,
            resample_id=resample,
        )
