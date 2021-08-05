# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format
"""

import os
import sys
import time
import pandas as pd

import sklearn.preprocessing
import sklearn.utils
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.dictionary_based import (
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
    WEASEL,
    MUSE,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    ProximityForest,
    ProximityTree,
    ProximityStump,
    KNeighborsTimeSeriesClassifier,
    ShapeDTW,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    MatrixProfileClassifier,
    SignatureClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.interval_based import (
    TimeSeriesForestClassifier,
    RandomIntervalSpectralForest,
    SupervisedTimeSeriesForest,
)
from sktime.classification.interval_based import CanonicalIntervalForest, DrCIF
from sktime.classification.kernel_based import ROCKETClassifier, Arsenal
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts
import sktime.datasets.tsc_dataset_names as dataset_lists
import sktime.utils.sampling

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!
import numpy as np

__author__ = ["Tony Bagnall"]

"""Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java

"""


def load_and_run_classification_experiment(
    problem_path,
    results_path,
    cls_name,
    dataset,
    classifier=None,
    resample_id=0,
    overwrite=False,
    format=".ts",
    train_file=False,
):
    """Run a classification experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    problem_path: Location of problem files, full path.
    results_path: Location of where to write results. Any required directories
        will be created
    cls_name: determines which classifier to use, as defined in set_classifier.
        This assumes predict_proba is
    implemented, to avoid predicting twice. May break some classifiers though
    dataset: Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+
                "_TRAIN"+format, same for "_TEST"
    resample_id: Seed for resampling. If set to 0, the default train/test split
                from file is used. Also used in output file name.
    overwrite: if set to False, this will only build results if there is not a
                result file already present. If
    True, it will overwrite anything already there
    format: Valid formats are ".ts", ".arff" and ".long".
    For more info on format, see   examples/Loading%20Data%20Examples.ipynb
    train_file: whether to generate train files or not. If true, it performs a
                10xCV on the train and saves
    """
    build_test = True
    if not overwrite:
        full_path = (
            str(results_path)
            + "/"
            + str(cls_name)
            + "/Predictions/"
            + str(dataset)
            + "/testFold"
            + str(resample_id)
            + ".csv"
        )
        if os.path.exists(full_path):
            print(
                full_path + " Already exists and overwrite false, not building Test."
            )  # noqua
            build_test = False
        if train_file:
            full_path = (
                str(results_path)
                + "/"
                + str(cls_name)
                + "/Predictions/"
                + str(dataset)
                + "/trainFold"
                + str(resample_id)
                + ".csv"
            )
            if os.path.exists(full_path):
                print(
                    full_path + " Already exists and overwrite set to false, "
                    "not building Train"
                )  # noqua
                train_file = False
        if train_file is False and build_test is False:
            return

    # TO DO: Automatically differentiate between problem types,
    # currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" + format)
    testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)
    if resample_id != 0:
        # allLabels = np.concatenate((trainY, testY), axis = None)
        # allData = pd.concat([trainX, testX])
        # train_size = len(trainY) / (len(trainY) + len(testY))
        # trainX, testX, trainY, testY = train_test_split(allData, allLabels,
        # train_size=train_size,
        # random_state=resampleID, shuffle=True,
        # stratify=allLabels)
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resample_id
        )

    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    if classifier is None:
        classifier = set_classifier(cls_name, resample_id)
    print(cls_name + " on " + dataset + " resample number " + str(resample_id))  # noqua
    if build_test:
        start = int(round(time.time() * 1000))
        classifier.fit(trainX, trainY)
        build_time = int(round(time.time() * 1000)) - start
        start = int(round(time.time() * 1000))
        probs = classifier.predict_proba(testX)
        preds = classifier.classes_[np.argmax(probs, axis=1)]
        test_time = int(round(time.time() * 1000)) - start
        ac = accuracy_score(testY, preds)
        print(
            cls_name
            + " on "
            + dataset
            + " resample number "
            + str(resample_id)
            + " test acc: "
            + str(ac)
            + " time: "
            + str(test_time)
        )  # noqua
        #        print(str(classifier.findEnsembleTrainAcc(trainX, trainY)))
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        temp = np.array_repr(classifier.classes_).replace("\n", "")

        third = (
            str(ac)
            + ","
            + str(build_time)
            + ","
            + str(test_time)
            + ",-1,-1,"
            + str(len(classifier.classes_))
        )  # noqua

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=preds,
            predicted_probs=probs,
            dataset_name=dataset,
            y_true=testY,
            split="TEST",
        )
    if train_file:
        start = int(round(time.time() * 1000))
        if build_test and hasattr(
            classifier, "_get_train_probs"
        ):  # Normally Can only do this if test has been built
            train_probs = classifier._get_train_probs(trainX)
        else:
            train_probs = cross_val_predict(
                classifier, X=trainX, y=trainY, cv=10, method="predict_proba"
            )
        train_time = int(round(time.time() * 1000)) - start
        train_preds = classifier.classes_[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(trainY, train_preds)
        print(
            cls_name
            + " on "
            + dataset
            + " resample number "
            + str(resample_id)
            + " train acc: "
            + str(train_acc)
            + " time: "
            + str(train_time)
        )  # noqua
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")
        temp = np.array_repr(classifier.classes_).replace("\n", "")
        third = (
            str(train_acc)
            + ","
            + str(train_time)
            + ",-1,-1,-1,"
            + str(len(classifier.classes_))
        )
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=train_preds,
            predicted_probs=train_probs,
            dataset_name=dataset,
            y_true=trainY,
            split="TRAIN",
        )


def run_classification_experiment(
    trainX,
    classifier,
    results_path,
    trainY,
    testX,
    testY=None,
    cls_name=None,
    dataset_name=None,
    resample_id=0,
    overwrite=False,
    train_file=False,
):
    """Run a classification experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    problem_path: Location of problem files, full path.
    results_path: Location of where to write results. Any required directories
        will be created
    cls_name: determines which classifier to use, as defined in set_classifier.
        This assumes predict_proba is
    implemented, to avoid predicting twice. May break some classifiers though
    dataset: Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+
                "_TRAIN"+format, same for "_TEST"
    resample_id: Seed for resampling. If set to 0, the default train/test split
                from file is used. Also used in output file name.
    overwrite: if set to False, this will only build results if there is not a
                result file already present. If
    True, it will overwrite anything already there
    format: Valid formats are ".ts", ".arff" and ".long".
    For more info on format, see   examples/Loading%20Data%20Examples.ipynb
    train_file: whether to generate train files or not. If true, it performs a
                10xCV on the train and saves
    """
    build_test = True
    if not overwrite:
        full_path = (
            str(results_path)
            + "/"
            + str(cls_name)
            + "/Predictions/"
            + str(dataset)
            + "/testFold"
            + str(resample_id)
            + ".csv"
        )
        if os.path.exists(full_path):
            build_test = False
        if train_file:
            full_path = (
                str(results_path)
                + "/"
                + str(cls_name)
                + "/Predictions/"
                + str(dataset)
                + "/trainFold"
                + str(resample_id)
                + ".csv"
            )
            if os.path.exists(full_path):
                train_file = False
        if train_file is False and build_test is False:
            return
    if resample_id != 0:
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resample_id
        )

    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    if classifier is None:
        classifier = set_classifier(cls_name, resample_id)
    print(cls_name + " on " + dataset + " resample number " + str(resample_id))
    if build_test:
        start = int(round(time.time() * 1000))
        classifier.fit(trainX, trainY)
        build_time = int(round(time.time() * 1000)) - start
        start = int(round(time.time() * 1000))
        probs = classifier.predict_proba(testX)
        preds = classifier.classes_[np.argmax(probs, axis=1)]
        test_time = int(round(time.time() * 1000)) - start
        ac = accuracy_score(testY, preds)
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        print(second)
        temp = np.array_repr(classifier.classes_).replace("\n", "")

        third = (
            str(ac)
            + ","
            + str(build_time)
            + ","
            + str(test_time)
            + ",-1,-1,"
            + str(len(classifier.classes_))
        )
        print(preds)
        print(type(preds))

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=preds,
            predicted_probs=probs,
            dataset_name=dataset,
            y_true=testY,
            split="TEST",
        )
    if train_file:
        start = int(round(time.time() * 1000))
        if build_test and hasattr(
            classifier, "_get_train_probs"
        ):  # Normally Can only do this if test has been built
            train_probs = classifier._get_train_probs(trainX)
        else:
            train_probs = cross_val_predict(
                classifier, X=trainX, y=trainY, cv=10, method="predict_proba"
            )
        train_time = int(round(time.time() * 1000)) - start
        train_preds = classifier.classes_[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(trainY, train_preds)
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")
        temp = np.array_repr(classifier.classes_).replace("\n", "")
        third = (
            str(train_acc)
            + ","
            + str(train_time)
            + ",-1,-1,-1,"
            + str(len(classifier.classes_))
        )
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=train_preds,
            predicted_probs=train_probs,
            dataset_name=dataset,
            y_true=trainY,
            split="TRAIN",
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
        load_and_run_classification_experiment(
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
        results_dir = ""
        classifier = "CIF"
        dataset = "UnitTest"
        resample = 0
        tf = False
        load_and_run_classification_experiment(
            overwrite=True,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=classifier,
            dataset=dataset,
            resample_id=resample,
            train_file=tf,
        )


def set_classifier(cls, resampleId=None):
    """Construct a classifier.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducability for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.

    Parameters
    ----------
    cls: String indicating which classifier you want
    resampleId: classifier random seed

    Return
    ------
    A BaseClassifier.
    """
    name = cls.lower()
    # Dictionary based
    if name == "boss" or name == "bossensemble":
        return BOSSEnsemble(random_state=resampleId)
    elif name == "cboss" or name == "contractableboss":
        return ContractableBOSS(random_state=resampleId)
    elif name == "tde" or name == "temporaldictionaryensemble":
        return TemporalDictionaryEnsemble(random_state=resampleId)
    elif name == "weasel":
        return WEASEL(random_state=resampleId)
    elif name == "muse":
        return MUSE(random_state=resampleId)
    # Distance based
    elif name == "pf" or name == "proximityforest":
        return ProximityForest(random_state=resampleId)
    elif name == "pt" or name == "proximitytree":
        return ProximityTree(random_state=resampleId)
    elif name == "ps" or name == "proximityStump":
        return ProximityStump(random_state=resampleId)
    elif name == "dtwcv" or name == "kneighborstimeseriesclassifier":
        return KNeighborsTimeSeriesClassifier(distance="dtwcv")
    elif name == "dtw" or name == "1nn-dtw":
        return KNeighborsTimeSeriesClassifier(distance="dtw")
    elif name == "msm" or name == "1nn-msm":
        return KNeighborsTimeSeriesClassifier(distance="msm")
    elif name == "ee" or name == "elasticensemble":
        return ElasticEnsemble()
    elif name == "shapedtw":
        return ShapeDTW()
    # Feature based
    elif name == "catch22":
        return Catch22Classifier(
            random_state=resampleId, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "matrixprofile":
        return MatrixProfileClassifier(random_state=resampleId)
    elif name == "signature":
        return SignatureClassifier(
            random_state=resampleId, classifier=RandomForestClassifier(n_estimators=500)
        )
    elif name == "tsfresh":
        return TSFreshClassifier(
            random_state=resampleId, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "tsfresh-r":
        return TSFreshClassifier(
            random_state=resampleId,
            estimator=RandomForestClassifier(n_estimators=500),
            relevant_feature_extractor=True,
        )
    # Hybrid
    elif name == "hivecotev1":
        return HIVECOTEV1(random_state=resampleId)
    # Interval based
    elif name == "rise" or name == "randomintervalspectralforest":
        return RandomIntervalSpectralForest(random_state=resampleId, n_estimators=500)
    elif name == "tsf" or name == "timeseriesforestclassifier":
        return TimeSeriesForestClassifier(random_state=resampleId, n_estimators=500)
    elif name == "cif" or name == "canonicalintervalforest":
        return CanonicalIntervalForest(random_state=resampleId, n_estimators=500)
    elif name == "stsf":
        return SupervisedTimeSeriesForest(random_state=resampleId, n_estimators=500)
    elif name == "drcif":
        return DrCIF(random_state=resampleId, n_estimators=500)
    # Kernel based
    elif name == "rocket":
        return ROCKETClassifier(random_state=resampleId)
    elif name == "arsenal":
        return Arsenal(random_state=resampleId)
    # Shapelet based
    elif name == "stc" or name == "shapelettransformclassifier":
        return ShapeletTransformClassifier(random_state=resampleId, n_estimators=500)
    elif name == "mrseql" or name == "mrseqlclassifier":
        return MrSEQLClassifier(seql_mode="fs", symrep=["sax", "sfa"])
    else:
        raise Exception("UNKNOWN CLASSIFIER")
