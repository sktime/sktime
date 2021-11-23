# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import time

import numpy as np

import sktime.datasets.tsc_dataset_names as dataset_lists
from sktime.benchmarking.experiments import load_and_run_classification_experiment
from sktime.classification.base import BaseClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.distances import distance, dtw_distance, euclidean_distance
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts

"""Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java.
"""


def _window_sizes():
    """Roadtest the new distances."""
    x = np.random.rand(1000)
    y = np.random.rand(1000)

    t = time.time()
    print("Full DTW Distance = ", dtw_distance(x, y), " takes = ", (time.time() - t))
    print(
        "Euclidean Distance = ",
        euclidean_distance(x, y),
        " takes = ",
        (time.time() - t),
    )
    print(
        "Using generalised distance function ED = ", distance(x, y, metric="euclidean")
    )
    print("Using generalised distance function DTW = ", distance(x, y, metric="dtw"))
    for w in range(0, len(x), 100):
        t = time.time()
        print(
            "Window ",
            w,
            " DTW Distance = ",
            dtw_distance(x, y, window=w),
            " takes = ",
            (time.time() - t),
        )
    for w in range(0, len(x), 100):
        t = time.time()
        print(
            "Window ",
            w,
            " DTW Distance = ",
            dtw_distance(x, y, lower_bounding=2, window=w),
            " takes = ",
            (time.time() - t),
        )


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


"""113 equal length/no missing univariate time series classification problems [3]"""
temp = [
    "Chinatown",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "Coffee",
    "Computers",
    "DiatomSizeReduction",
    "Earthquakes",
    "ECG200",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MoteStrain",
    "OliveOil",
    "OSULeaf",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "ShapesAll",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]
if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    _window_sizes()
    if sys.argv.__len__() > 1:  # cluster run, this is fragile
        print(sys.argv)
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        classifier = sys.argv[3]
        dataset = sys.argv[4]
        resample = int(sys.argv[5]) - 1

        if len(sys.argv) > 6:
            tf = sys.argv[6].lower() == "true"
        else:
            tf = False

        if len(sys.argv) > 7:
            predefined_resample = sys.argv[7].lower() == "true"
        else:
            predefined_resample = False

        load_and_run_classification_experiment(
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=classifier,
            dataset=dataset,
            resample_id=resample,
            build_train=tf,
            predefined_resample=predefined_resample,
        )
    else:  # Local run
        print(" Local Run")
        data_dir = "Z:/ArchiveData/Univariate_ts/"
        results_dir = "Z:/Results Working Area/Debug/DistancesNew/"
        classifier = "1nn-dtw"
        # dataset = "UnitTest"
        resample = 0
        tf = False
        predefined_resample = False
        for i in range(0, len(temp)):
            print(" Running problem ", temp[i])
            load_and_run_classification_experiment(
                overwrite=True,
                problem_path=data_dir,
                results_path=results_dir,
                cls_name=classifier,
                classifier=KNeighborsTimeSeriesClassifier(metric="dtw"),
                dataset=temp[i],
                resample_id=resample,
                build_train=tf,
                predefined_resample=predefined_resample,
            )
