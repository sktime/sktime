# -*- coding: utf-8 -*-
""" experiments.py: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results
are written in a standard format
TO DO: Tidy up this file!
"""

import os

import sklearn.preprocessing
import sklearn.utils

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
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid._catch22_forest_classifier import (
    Catch22ForestClassifier,
)
from sktime.classification.interval_based import (
    TimeSeriesForest,
    RandomIntervalSpectralForest,
)
from sktime.classification.interval_based._cif import CanonicalIntervalForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.shapelet_based import MrSEQLClassifier, ROCKETClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier


os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
import time
import numpy as np
import pandas as pd


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts
import sktime.datasets.tsc_dataset_names as dataset_lists

__author__ = ["Tony Bagnall"]

""" Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java

"""

classifier_list = [
    # Distance based
    "ProximityForest",
    "KNeighborsTimeSeriesClassifier",
    "ElasticEnsemble",
    "ShapeDTW",
    # Dictionary based
    "BOSS",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "WEASEL",
    "MUSE",
    # Interval based
    "RandomIntervalSpectralForest",
    "TimeSeriesForest",
    "CanonicalIntervalForest",
    # Shapelet based
    "ShapeletTransformClassifier",
    "ROCKET",
    "MrSEQLClassifier",
]


def set_classifier(cls, resampleId=None):
    """
    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducability. You can set up bespoke classifier in many other ways.

    :param cls: String indicating which classifier you want
    :param resampleId: classifier random seed

    :return: A classifier.

    """
    name = cls.lower()
    # Distance based
    if name == "pf" or name == "proximityforest":
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
    # Dictionary based
    elif name == "boss" or name == "bossensemble":
        return BOSSEnsemble(random_state=resampleId)
    elif name == "cboss" or name == "contractableboss":
        return ContractableBOSS(random_state=resampleId)
    elif name == "tde" or name == "temporaldictionaryensemble":
        return TemporalDictionaryEnsemble(random_state=resampleId)
    elif name == "weasel":
        return WEASEL(random_state=resampleId)
    elif name == "muse":
        return MUSE(random_state=resampleId)
    # Interval based
    elif name == "rise" or name == "randomintervalspectralforest":
        return RandomIntervalSpectralForest(random_state=resampleId)
    elif name == "tsf" or name == "timeseriesforest":
        return TimeSeriesForest(random_state=resampleId)
    elif name == "cif" or name == "canonicalintervalforest":
        return CanonicalIntervalForest(random_state=resampleId)
    elif name == "drcif":
        return DrCIF(random_state=resampleId)
    # Shapelet based
    elif name == "stc" or name == "shapelettransformclassifier":
        return ShapeletTransformClassifier(
            random_state=resampleId, time_contract_in_mins=1
        )
    elif name == "mrseql" or name == "mrseqlclassifier":
        return MrSEQLClassifier(seql_mode="fs", symrep=["sax", "sfa"])
    elif name == "rocket":
        return ROCKETClassifier(random_state=resampleId)
    # Hybrid
    elif name == "catch22":
        return Catch22ForestClassifier(random_state=resampleId)
    elif name == "hivecotev1":
        return HIVECOTEV1(random_state=resampleId)
    else:
        raise Exception("UNKNOWN CLASSIFIER")


def stratified_resample(X_train, y_train, X_test, y_test, random_state):
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = pd.concat([X_train, X_test])
    random_state = sklearn.utils.check_random_state(random_state)
    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    assert list(unique_train) == list(
        unique_test
    )  # haven't built functionality to deal with classes that exist in
    # test but not in train
    # prepare outputs
    X_train = pd.DataFrame()
    y_train = np.array([])
    X_test = pd.DataFrame()
    y_test = np.array([])
    # for each class
    for label_index in range(0, len(unique_train)):
        # derive how many instances of this class from the counts
        num_instances = counts_train[label_index]
        # get the indices of all instances with this class label
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        # shuffle them
        random_state.shuffle(indices)
        # take the first lot of instances for train, remainder for test
        train_indices = indices[0:num_instances]
        test_indices = indices[num_instances:]
        del indices  # just to make sure it's not used!
        # extract data from corresponding indices
        train_instances = all_data.iloc[train_indices, :]
        test_instances = all_data.iloc[test_indices, :]
        train_labels = all_labels[train_indices]
        test_labels = all_labels[test_indices]
        # concat onto current data from previous loop iterations
        X_train = pd.concat([X_train, train_instances])
        X_test = pd.concat([X_test, test_instances])
        y_train = np.concatenate([y_train, train_labels], axis=None)
        y_test = np.concatenate([y_test, test_labels], axis=None)
    # get the counts of the new train and test resample
    unique_train_new, counts_train_new = np.unique(y_train, return_counts=True)
    unique_test_new, counts_test_new = np.unique(y_test, return_counts=True)
    # make sure they match the original distribution of data
    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)
    return X_train, y_train, X_test, y_test


def run_experiment(
    problem_path,
    results_path,
    cls_name,
    dataset,
    classifier=None,
    resampleID=0,
    overwrite=False,
    format=".ts",
    train_file=False,
):
    """
    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.
    :param problem_path: Location of problem files, full path.
    :param results_path: Location of where to write results. Any required directories
        will be created
    :param cls_name: determines which classifier to use, as defined in set_classifier.
        This assumes predict_proba is
    implemented, to avoid predicting twice. May break some classifiers though
    :param dataset: Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+
                "_TRAIN"+format, same for "_TEST"
    :param resampleID: Seed for resampling. If set to 0, the default train/test split
                from file is used. Also used in output file name.
    :param overwrite: if set to False, this will only build results if there is not a
                result file already present. If
    True, it will overwrite anything already there
    :param format: Valid formats are ".ts", ".arff" and ".long".
    For more info on format, see   examples/Loading%20Data%20Examples.ipynb
    :param train_file: whether to generate train files or not. If true, it performs a
                10xCV on the train and saves
    :return:
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
            + str(resampleID)
            + ".csv"
        )
        if os.path.exists(full_path):
            print(
                full_path
                + " Already exists and overwrite set to false, not building Test"
            )
            build_test = False
        if train_file:
            full_path = (
                str(results_path)
                + "/"
                + str(cls_name)
                + "/Predictions/"
                + str(dataset)
                + "/trainFold"
                + str(resampleID)
                + ".csv"
            )
            if os.path.exists(full_path):
                print(
                    full_path
                    + " Already exists and overwrite set to false, not building Train"
                )
                train_file = False
        if train_file == False and build_test == False:
            return

    # TO DO: Automatically differentiate between problem types,
    # currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" + format)
    testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)
    if resampleID != 0:
        # allLabels = np.concatenate((trainY, testY), axis = None)
        # allData = pd.concat([trainX, testX])
        # train_size = len(trainY) / (len(trainY) + len(testY))
        # trainX, testX, trainY, testY = train_test_split(allData, allLabels,
        # train_size=train_size,
        # random_state=resampleID, shuffle=True,
        # stratify=allLabels)
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resampleID
        )

    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    if classifier is None:
        classifier = set_classifier(cls_name, resampleID)
    print(cls_name + " on " + dataset + " resample number " + str(resampleID))
    if build_test:
        # TO DO : use sklearn CV
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
            + str(resampleID)
            + " test acc: "
            + str(ac)
            + " time: "
            + str(test_time)
        )
        #        print(str(classifier.findEnsembleTrainAcc(trainX, trainY)))
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
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            classifier_name=cls_name,
            resample_seed=resampleID,
            predicted_class_vals=preds,
            actual_probas=probs,
            dataset_name=dataset,
            actual_class_vals=testY,
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
            + str(resampleID)
            + " train acc: "
            + str(train_acc)
            + " time: "
            + str(train_time)
        )
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
            classifier_name=cls_name,
            resample_seed=resampleID,
            predicted_class_vals=train_preds,
            actual_probas=train_probs,
            dataset_name=dataset,
            actual_class_vals=trainY,
            split="TRAIN",
        )


def write_results_to_uea_format(
    output_path,
    classifier_name,
    dataset_name,
    actual_class_vals,
    predicted_class_vals,
    split="TEST",
    resample_seed=0,
    actual_probas=None,
    second_line="No Parameter Info",
    third_line="N/A",
    class_labels=None,
):
    """
    This is very alpha and I will probably completely change the structure once train
    fold is sorted, as that internally
    does all this I think!
    Output mirrors that produced by this Java
    :param output_path:
    :param classifier_name:
    :param dataset_name:
    :param actual_class_vals:
    :param predicted_class_vals:
    :param split:
    :param resample_seed:
    :param actual_probas:
    :param second_line:
    :param third_line:
    :param class_labels:
    :return:
    """
    if len(actual_class_vals) != len(predicted_class_vals):
        raise IndexError(
            "The number of predicted class values is not the same as the "
            + "number of actual class values"
        )

    try:
        os.makedirs(
            str(output_path)
            + "/"
            + str(classifier_name)
            + "/Predictions/"
            + str(dataset_name)
            + "/"
        )
    except os.error:
        pass  # raises os.error if path already exists

    if split == "TRAIN" or split == "train":
        train_or_test = "train"
    elif split == "TEST" or split == "test":
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")

    file = open(
        str(output_path)
        + "/"
        + str(classifier_name)
        + "/Predictions/"
        + str(dataset_name)
        + "/"
        + str(train_or_test)
        + "Fold"
        + str(resample_seed)
        + ".csv",
        "w",
    )

    # <classifierName>,<datasetName>,<train/test>,<Class Labels>
    file.write(
        str(dataset_name)
        + ","
        + str(classifier_name)
        + ","
        + str(train_or_test)
        + ","
        + str(resample_seed)
        + ",MILLISECONDS,PREDICTIONS, Generated by experiments.py"
    )
    file.write("\n")

    # the second line of the output is free form and classifier-specific;
    # usually this will record info
    # such as parameter options used, any constituent model names for ensembles, etc.
    file.write(str(second_line) + "\n")

    # the third line of the file is the accuracy (should be between 0 and 1 inclusive).
    # If this is a train output file then it will be a training estimate of the
    # classifier on the training data only (e.g. #10-fold cv, leave-one-out cv, etc.).
    # If this is a test output file, it should be the output of the estimator on the
    # test data (likely trained on the training data for a-priori para optimisation)
    file.write(str(third_line))
    file.write("\n")

    # from line 4 onwards each line should include the actual and predicted class labels
    # (comma-separated). If present, for each case, the probabilities of predicting
    # every class value for this case should also be appended to the line (a space is
    # also included between the predicted value and the predict_proba). E.g.:
    # if predict_proba data IS provided for case i:
    #   actual_class_val[i], predicted_class_val[i],,
    #
    # if predict_proba data IS NOT provided for case i:
    #   actual_class_val[i], predicted_class_val[i]
    for i in range(0, len(predicted_class_vals)):
        file.write(str(actual_class_vals[i]) + "," + str(predicted_class_vals[i]))
        if actual_probas is not None:
            file.write(",")
            for j in actual_probas[i]:
                file.write("," + str(j))
            file.write("\n")

    file.close()


def test_loading():

    # test multivariate
    # Test univariate
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


benchmark_datasets = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Ham",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "SmallKitchenAppliances",
    "SmoothSubspace",
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
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]


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
        run_experiment(
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=classifier,
            dataset=dataset,
            resampleID=resample,
            train_file=tf,
        )
    else:  # Local run
        print(" Local Run")
        data_dir = "Z:/ArchiveData/Univariate_ts/"
        results_dir = "Z:/Results Working Area/DistanceBased/sktime/"
        dataset = "ArrowHead"
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        classifier = "1NN-MSM"
        resample = 0
        #         for i in range(0, len(univariate_datasets)):
        #             dataset = univariate_datasets[i]
        # #            print(i)
        # #            print(" problem = "+dataset)
        tf = False
        for i in range(0, len(benchmark_datasets)):
            dataset = benchmark_datasets[i]
            run_experiment(
                overwrite=True,
                problem_path=data_dir,
                results_path=results_dir,
                cls_name=classifier,
                dataset=dataset,
                resampleID=resample,
                train_file=tf,
            )
