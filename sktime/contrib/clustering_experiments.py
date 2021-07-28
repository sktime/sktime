# -*- coding: utf-8 -*-
__all__ = ["load_and_run_clustering_experiment"]


""" Cluster Experiments
code to run experiments for clustering, saving results in a standard format.

The main method is run_clustering_experiment. However, this file is also configured for 
runs of the main method with command line arguments, or for single debugging runs. 
"""

import os
import sklearn.preprocessing
import sklearn.utils
from sktime.clustering import (
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
)
from sktime.utils.data_io import write_results_to_uea_format

# We sometimes want to force execution in a single thread. sklearn often threads in ways
# beyond the users control. This forces single thread execution, which is required,
# for example, when running on an HPC
# MUST be done before numpy import
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


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


def run_clustering_experiment(
    trainX,
    clusterer,
    results_path,
    trainY=None,
    testX=None,
    testY=None,
    cls_name=None,
    dataset_name=None,
    resampleID=0
):
    """
    Run a clustering experiment and save the results to file.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv. This
    version loads the data from file based on a path. The clusterer is always trained on
    the required input data trainX. Output to trainResample<resampleID>.csv will be
    the predicted clusters of trainX. If trainY is also passed, these are written to
    file. If the clusterer makes probabilistic predictions, these are also written to
    file. See write_results_to_uea_format for more on the output. Be warned,
    this method will always overwrite existing results, check bvefore calling or use
    load_and_run_clustering_experiment instead.

    Parameters
    ----------
    trainX : pd.DataFrame or np.array
        The data to cluster.
    clusterer : BaseClusterer
        The clustering object
    results_path : str
    trainY : np.array, default = None
        Train data tue class labels, only used for file writing, ignored by the
        clusterer
    testX : pd.DataFrame or np.array, default = None
        Test attribute data, if present it is used for predicting testY
    testY : np.array, default = None
        Test data true class labels, only used for file writing, ignored by the
        clusterer
    cls_name : str, default = None
        Name of the clusterer, written to the results file, ignored if None
    dataset_name : str, default = None
        Name of problem, written to the results file, ignored if null
    resampleID : int, default = 0
        Resample identifier, defaults to 0
    """
    # Build the clusterer on train data, recording how long it takes
    start = int(round(time.time() * 1000))
    clusterer.fit(trainX)
    build_time = int(round(time.time() * 1000)) - start
    start = int(round(time.time() * 1000))
    train_preds = clusterer.predict(trainX)
    predict_train_time = int(round(time.time() * 1000)) - start

    # Form predictions on trainY
    start = int(round(time.time() * 1000))
    preds = clusterer.predict(testX)
    second = str(clusterer.get_params())
    second.replace("\n", " ")
    second.replace("\r", " ")
    # TODO: refactor clusterers to return an array
    pr = np.array(preds)
    third = "," + str(build_time) + "," + str(test_time) + ",-1,-1,"
    write_results_to_uea_format(
        second_line=second,
        third_line=third,
        output_path=results_path,
        estimator_name=cls_name,
        resample_seed=resampleID,
        y_pred=preds,
        dataset_name=dataset_name,
        y_true=testY,
        split="TEST",
    )

    #        preds = form_cluster_list(clusters, len(testY))
    test_time = int(round(time.time() * 1000)) - start
    #        print(str(classifier.findEnsembleTrainAcc(trainX, trainY)))
    if "Composite" in cls_name:
        second = "Para info too long!"
    else:

        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(clusterer.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")
        third = "FORMAT NOT FINALISED"
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resampleID,
            y_pred=train_preds,
            dataset_name=dataset_name,
            y_true=trainY,
            split="TRAIN",
        )

def load_and_run_clustering_experiment(
    problem_path,
    results_path,
    cls_name,
    dataset,
    clusterer=None,
    resampleID=0,
    overwrite=False,
    format=".ts",
    train_file=False,
):
    """Run a clustering experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv. This
    version loads the data from file based on a path. The
    clusterer is always trained on the

    Parameters
    ----------
    problem_path: Location of problem files, full path.
    results_path: Location of where to write results. Any required directories
        will be created
    cls_name: determines which clusterer to use, as defined in set_classifier.
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

    #Set up the file path in standard format
    if not overwrite:
        full_path = (
            str(results_path)
            + "/"
            + str(cls_name)
            + "/Predictions/"
            + str(dataset)
            + "/testResample"
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
                + "/trainResample"
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

    # currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" + format)
    testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)
    if resampleID != 0:
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resampleID
        )
    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    if clusterer is None:
        clusterer = set_clusterer(cls_name, resampleID)



    #run_clustering_experiment(trainX,trainY,testX,testY,results_path,cls_name,
                              dataset,clusterer, resampleID,overwrite,train_file)

    print(cls_name + " on " + dataset + " resample number " + str(resampleID))




    # Build the clusterer on train
    start = int(round(time.time() * 1000))
    clusterer.fit(trainX)
    build_time = int(round(time.time() * 1000)) - start
    start = int(round(time.time() * 1000))
    preds = clusterer.predict(testX)
    #        preds = form_cluster_list(clusters, len(testY))
    test_time = int(round(time.time() * 1000)) - start
    print(
        cls_name
        + " on "
        + dataset
        + " resample number "
        + str(resampleID)
        + " time: "
        + str(test_time)
    )
    #        print(str(classifier.findEnsembleTrainAcc(trainX, trainY)))
    if "Composite" in cls_name:
        second = "Para info too long!"
    else:
        second = str(clusterer.get_params())
    second.replace("\n", " ")
    second.replace("\r", " ")
    # TODO: refactor clusterers to return an array
    pr = np.array(preds)
    third = "," + str(build_time) + "," + str(test_time) + ",-1,-1,"
    write_results_to_uea_format(
        second_line=second,
        third_line=third,
        output_path=results_path,
        estimator_name=cls_name,
        resample_seed=resampleID,
        y_pred=preds,
        dataset_name=dataset,
        y_true=testY,
        split="TEST",
    )
    if train_file:
        start = int(round(time.time() * 1000))
        train_preds = clusterer.predict(trainX)
        train_time = int(round(time.time() * 1000)) - start
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(clusterer.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")
        third = "FORMAT NOT FINALISED"
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resampleID,
            y_pred=train_preds,
            dataset_name=dataset,
            y_true=trainY,
            split="TRAIN",
        )


def set_clusterer(cls, resampleId=None):
    """Construct a clusterer.

    Basic way of creating the clusterer to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducability through run_clustering_experiment. You can set up bespoke
    clusterers and pass them to run_clustering_experiment if you prefer. It also
    serves to illustrate the base clusterer parameters

    Parameters
    ----------
    cls: String indicating which clusterer you want
    resampleId: classifier random seed

    Return
    ------
    A clusterer.
    """
    name = cls.lower()
    # Distance based
    if name == "kmeans" or name == "k-means":
        return TimeSeriesKMeans(
            n_clusters=5,
            max_iter=50,
            averaging_algorithm="mean",
            random_state=resampleId,
        )
    if name == "kmedoids" or name == "k-medoids":
        return TimeSeriesKMedoids(
            n_clusters=5,
            max_iter=50,
            averaging_algorithm="mean",
            random_state=resampleId,
        )

    else:
        raise Exception("UNKNOWN CLUSTERER")


def stratified_resample(X_train, y_train, X_test, y_test, random_state):
    """Resample data using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new trrain and test.

    Parameters
    ----------
    X_train: train data attributes in sktime pandas format.
    y_train: train data class labes as np array.
    X_test: test data attributes in sktime pandas format.
    y_test: test data class labes as np array.

    Returns
    -------
    new train and test attributes and class labels.
    """
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
            resampleID=resample,
            train_file=tf,
        )
    else:  # Local run
        print(" Local Run")
        data_dir = "../datasets/data/"
        results_dir = "C:/Temp/Clusterers/"
        dataset = "UnitTest"
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        classifier = "kmeans"
        resample = 0
        tf = True
        load_and_run_clustering_experiment(
            overwrite=True,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=classifier,
            dataset=dataset,
            resampleID=resample,
            train_file=tf,
        )
