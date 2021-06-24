# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sktime.datasets.base import load_UCR_UEA_dataset
from sktime.classification.kernel_based import ROCKETClassifier
from sktime.utils.data_io import write_results_to_uea_format


def run_single_experiment(
    trainX,
    trainY,
    testX,
    testY,
    results_path,
    cls_name,
    dataset,
    classifier=None,
    resampleID=0,
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
    resampleID: Seed for resampling. If set to 0, the default train/test split
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

    if resampleID != 0:
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


"""
List formatted EEG classification problems available on timeseriesclassification.com"""
problem_list = [
    "EyesOpenShut",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP1",
    ]


if __name__ == "__main__":
    """
    Example simple usage, with data loaded from tsc.com
    """
    problem="SelfRegulationSCP1"
    trainX,trainY=load_UCR_UEA_dataset(problem,split="train",return_X_y=True)
    testX,testY=load_UCR_UEA_dataset(problem,split="test",return_X_y=True)
    rc= ROCKETClassifier()
    run_single_experiment(trainX,trainY,testX,testY,results_path="C:/temp/",
    cls_name="Rocket",dataset=problem,classifier=rc)


