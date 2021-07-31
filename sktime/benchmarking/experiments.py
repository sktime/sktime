# -*- coding: utf-8 -*-
"""Functions to perform classification and clustering experiments.

Results are saved a standardised format used by both tsml and sktime.
"""
__author__ = ["Tony Bagnall"]
__all__ = [
    "run_clustering_experiment",
    "load_and_run_clustering_experiment",
    "set_clusterer",
]


import os
import time
import numpy as np

from sklearn import preprocessing
from sktime.utils.data_io import write_results_to_uea_format
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts
from sktime.utils.sampling import stratified_resample
from sktime.clustering import (
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
)


def run_clustering_experiment(
    trainX,
    clusterer,
    results_path,
    trainY=None,
    testX=None,
    testY=None,
    cls_name=None,
    dataset_name=None,
    resample_id=0,
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
        Where to write the results to
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
        Name of problem, written to the results file, ignored if None
    resample_id : int, default = 0
        Resample identifier, defaults to 0

    """
    # Build the clusterer on train data, recording how long it takes

    start = int(round(time.time() * 1000))
    clusterer.fit(trainX)
    build_time = int(round(time.time() * 1000)) - start
    start = int(round(time.time() * 1000))
    train_preds = clusterer.predict(trainX)
    # predict_train_time = int(round(time.time() * 1000)) - start

    # Form predictions on trainY
    start = int(round(time.time() * 1000))
    preds = clusterer.predict(testX)
    test_time = int(round(time.time() * 1000)) - start
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
        resample_seed=resample_id,
        y_pred=pr,
        dataset_name=dataset_name,
        y_true=testY,
        split="TEST",
        full_path=False,
    )

    #        preds = form_cluster_list(clusters, len(testY))
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
            resample_seed=resample_id,
            y_pred=train_preds,
            dataset_name=dataset_name,
            y_true=trainY,
            split="TRAIN",
            full_path=False,
        )


def load_and_run_clustering_experiment(
    problem_path,
    results_path,
    cls_name,
    dataset,
    clusterer=None,
    resample_id=0,
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
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created
    cls_name : str
        determines which clusterer to use if clusterer is None. In this
        case, set_clusterer is called with this cls_name
    dataset : str
        Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+
        "_TRAIN"+format, same for "_TEST"
    resample_id : int, default = 0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    overwrite : boolean, default = False
        if False, this will only build results if there is not a result file already
        present. If True, it will overwrite anything already there.
    format: string, default = ".ts"
        Valid formats are ".ts", ".arff", ".tsv" and ".long". For more info on
        format, see   examples/loading_data.ipynb
    train_file: boolean, default = False
        whether to generate train files or not. If true, it performs a 10xCV on the
        train and saves
    """
    # Set up the file path in standard format
    if not overwrite:
        full_path = (
            str(results_path)
            + "/"
            + str(cls_name)
            + "/Predictions/"
            + str(dataset)
            + "/testResample"
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
                + "/trainResample"
                + str(resample_id)
                + ".csv"
            )
            if os.path.exists(full_path):
                train_file = False
        if train_file is False and build_test is False:
            return

    # currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" + format)
    testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)
    if resample_id != 0:
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resample_id
        )
    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    if clusterer is None:
        clusterer = set_clusterer(cls_name, resample_id)

    run_clustering_experiment(
        trainX,
        clusterer,
        trainY=trainY,
        testX=testX,
        testY=testY,
        cls_name=cls_name,
        dataset_name=dataset,
        results_path=results_path,
    )


def set_clusterer(cls, resample_id=None):
    """Construct a clusterer.

    Basic way of creating the clusterer to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducability through run_clustering_experiment. You can set up bespoke
    clusterers and pass them to run_clustering_experiment if you prefer. It also
    serves to illustrate the base clusterer parameters

    Parameters
    ----------
    cls : str
        indicating which clusterer you want
    resample_id : int or None, default = None
        clusterer random seed

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
            random_state=resample_id,
        )
    if name == "kmedoids" or name == "k-medoids":
        return TimeSeriesKMedoids(
            n_clusters=5,
            max_iter=50,
            averaging_algorithm="mean",
            random_state=resample_id,
        )

    else:
        raise Exception("UNKNOWN CLUSTERER")
