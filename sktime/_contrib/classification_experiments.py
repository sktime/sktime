"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard results format.
"""

__author__ = ["TonyBagnall"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sktime.datasets.tsc_dataset_names as dataset_lists
from sktime._contrib.set_classifier import set_classifier
from sktime.benchmarking.experiments import load_and_run_classification_experiment
from sktime.classification.deep_learning import CNNClassifier
from sktime.classification.feature_based import FreshPRINCE
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.datasets import load_from_tsfile_to_dataframe as load_ts

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


if __name__ == "__main__":
    """Example simple usage, with arguments input via script or hard coded for
    testing."""
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
            classifier=set_classifier(classifier, resample, tf),
            cls_name=classifier,
            dataset=dataset,
            resample_id=resample,
            build_train=tf,
            predefined_resample=predefined_resample,
        )
    else:  # Local run
        print(" Local Run")
        data_dir = "../datasets/data/"
        results_dir = "./temp/"
        cls_name = "FreshPRINCE"
        classifier = FreshPRINCE()
        dataset = "UnitTest"
        resample = 0
        tf = False
        predefined_resample = False

        load_and_run_classification_experiment(
            overwrite=True,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=cls_name,
            classifier=classifier,
            dataset=dataset,
            resample_id=resample,
            build_train=tf,
            predefined_resample=predefined_resample,
        )
