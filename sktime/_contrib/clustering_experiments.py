"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os
import sys

import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!
import sklearn.metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

import sktime.datasets.tsc_dataset_names as dataset_lists
from sktime.benchmarking.experiments import run_clustering_experiment
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_from_tsfile as load_ts

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


def config_clusterer(clusterer: str, **kwargs):
    """Config clusterer."""
    if clusterer == "kmeans":
        cls = TimeSeriesKMeans(**kwargs)
    elif clusterer == "kmedoids":
        cls = TimeSeriesKMedoids(**kwargs)
    return cls


def tune_window(metric: str, train_X):
    """Tune window."""
    best_w = 0
    best_score = 0
    for w in np.arange(0, 1, 0.1):
        cls = TimeSeriesKMeans(metric=metric, distance_params={"window": w})
        cls.fit(train_X)
        preds = cls.predict(train_X)
        print(" Preds type = ", type(preds))
        score = davies_bouldin_score(train_X, preds)
        print(score)
        if score > best_score:
            best_score = score
            best_w = w
    print("best window =", best_w, " with score ", best_score)
    return best_w


if __name__ == "__main__":
    """Example simple usage, with arguments input via script or hard coded for
    testing."""
    clusterer = "kmeans"
    chris_config = True  # This is so chris doesn't have to change config each time
    tune = False

    if sys.argv.__len__() > 1:  # cluster run, this is fragile
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        distance = sys.argv[3]
        dataset = sys.argv[4]
        resample = int(sys.argv[5]) - 1
        tf = True
    # kraken run, needs sorting out
    #        print(sys.argv)
    #        data_dir = "/home/ajb/data/Univariate_ts/"
    #        results_dir = "/home/ajb/results/kmeans/"
    #        dataset = sys.argv[1]
    #        resample = int(sys.argv[2]) - 1
    #        tf = True
    #        distance = sys.argv[3]
    elif chris_config is True:
        path = "C:/Users/chris/Documents/Masters"
        data_dir = os.path.abspath(f"{path}/datasets/Univariate_ts/")
        results_dir = os.path.abspath(f"{path}/results/")
        dataset = "ElectricDevices"
        resample = 2
        tf = True
        distance = "msm"
    else:  # Local run
        print(" Local Run")
        dataset = "ElectricDevices"
        data_dir = f"../datasets/data/"
        results_dir = "./temp"
        resample = 0
        tf = True
        distance = "msm"
    train_X, train_Y = load_ts(
        f"{data_dir}/{dataset}/{dataset}_TRAIN.ts", return_data_type="numpy2d"
    )
    test_X, test_Y = load_ts(
        f"{data_dir}/{dataset}/{dataset}_TEST.ts", return_data_type="numpy2d"
    )
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler()
    train_X = s.fit_transform(train_X.T)
    train_X = train_X.T
    test_X = s.fit_transform(test_X.T)
    test_X = test_X.T
    if tune:
        window = tune_window(distance, train_X)
        name = clusterer + "-" + distance + "-tuned"
    else:
        name = clusterer + "-" + distance
    if (
        distance == "wdtw"
        or distance == "dwdtw"
        or distance == "dtw"
        or distance == "wdtw"
    ):
        parameters = {"window": 0.2, "epsilon": 0.05, "g": 0.05, "c": 1}
    else:
        parameters = {"window": 1.0, "epsilon": 0.05, "g": 0.05, "c": 1}
    clst = TimeSeriesKMeans(
        averaging_method="dba",
        average_params={"averaging_distance_metric": distance},
        metric=distance,
        distance_params=parameters,
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
        cls_name=name,
        dataset_name=dataset,
        resample_id=resample,
        overwrite=True,
    )
    print("done")
