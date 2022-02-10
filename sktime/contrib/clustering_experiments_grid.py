# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV

from sktime.contrib.set_classifier import set_classifier

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sktime.datasets.tsc_dataset_names as dataset_lists
from sktime.benchmarking.experiments import run_clustering_experiment
from sktime.clustering import TimeSeriesKMeans, TimeSeriesKMedoids
from sktime.datasets import load_from_tsfile as load_ts

"""Prototype mechanism for testing clusterers on the UCR format. This mirrors the
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


def config_clusterer(clusterer: str, **kwargs):
    """Configure the custerer for experiments."""
    if clusterer == "kmeans":
        cls = TimeSeriesKMeans(**kwargs)
    elif clusterer == "kmedoids":
        cls = TimeSeriesKMedoids(**kwargs)
    return cls


def _get_bounding_matrix_params():
    range = np.linspace(0, 1, 11)
    param_names = ["window", "itakura_max_slope"]
    hyper_params = {}
    hyper_params["metric"] = ["dtw"]
    hyper_params["distance_params"] = []
    for param in param_names:
        for val in range:
            hyper_params["distance_params"].append({param: val})

    return hyper_params


def hyper_param_experiment(clusterer: str):
    """Hyper parametrise a clusters."""
    params = _get_bounding_matrix_params()
    # params.append({"metric": ["euclidean"]})
    return params
    # for param in params:
    #     yield config_clusterer(clusterer, **param)


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    hyperparams = True  # Set to true to enable running hyper params
    clusterer = "kmeans"
    chris_config = False  # This is so chris doesn't have to change config each time

    if sys.argv.__len__() > 1:  # cluster run, this is fragile
        print(sys.argv)
        data_dir = "/home/ajb/data/Univariate_ts/"
        results_dir = "/home/ajb/results/"
        dataset = sys.argv[1]
        resample = int(sys.argv[2]) - 1
        tf = True
        distance = sys.argv[3]
    elif chris_config is True:
        path = "/home/chris/Documents/masters-results/"
        data_dir = os.path.abspath(f"{path}/datasets/")
        results_dir = os.path.abspath(f"{path}/results/")
        dataset = "GunPoint"
        resample = 2
        tf = True
        distance = "euclidean"
    else:  # Local run
        print(" Local Run")
        data_dir = "Z:/ArchiveData/Univariate_ts/"
        results_dir = "./temp"
        dataset = "GunPoint"
        resample = 22
        tf = True
        distance = "euclidean"

    # train_X, train_Y = load_ts(f"{data_dir}/{dataset}/{dataset}_TRAIN.ts")
    # test_X, test_Y = load_ts(f"{data_dir}/{dataset}/{dataset}_TEST.ts")
    train_X, train_Y = load_ts(
        f"{data_dir}/{dataset}/{dataset}_TRAIN.ts", return_data_type="numpy2d"
    )
    test_X, test_Y = load_ts(
        f"{data_dir}/{dataset}/{dataset}_TEST.ts", return_data_type="numpy2d"
    )
    print(" input type = ", type(test_X))

    if hyperparams is True:
        hyper_param_clusterers = hyper_param_experiment(clusterer)
        # Comment this out onyl meant to be used to run it quickly
        hyper_param_clusterers = {"metric": ["dtw"]}
        clus = GridSearchCV(TimeSeriesKMeans(), hyper_param_clusterers, verbose=True)
        clus.fit(train_X)
        clus.predict(test_X)
        stop = ""
    else:
        clst = config_clusterer(
            clusterer=clusterer,
            metric=distance,
            n_clusters=len(set(train_Y)),
            random_state=resample + 1,
        )
        run_clustering_experiment(
            train_X,
            clst,
            results_path=results_dir,
            trainY=train_Y,
            testX=test_X,
            testY=test_Y,
            cls_name=clusterer + "_" + distance,
            dataset_name=dataset,
            resample_id=resample,
        )
