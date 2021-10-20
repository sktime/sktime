# -*- coding: utf-8 -*-
"""Functions to perform classification and clustering experiments.

Results are saved a standardised format used by both tsml and sktime.
"""
__author__ = ["Tony Bagnall"]
__all__ = [
    "run_clustering_experiment",
    "load_and_run_clustering_experiment",
    "set_clusterer",
    "run_classification_experiment",
    "load_and_run_classification_experiment",
    "set_classifier",
]


import os
import time
from datetime import datetime

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sktime.classification.dictionary_based import (
    MUSE,
    WEASEL,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    KNeighborsTimeSeriesClassifier,
    ProximityForest,
    ProximityStump,
    ProximityTree,
    ShapeDTW,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    MatrixProfileClassifier,
    SignatureClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralForest,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import Arsenal, ROCKETClassifier
from sktime.classification.shapelet_based import (
    MrSEQLClassifier,
    ShapeletTransformClassifier,
)
from sktime.clustering import TimeSeriesKMeans, TimeSeriesKMedoids
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts
from sktime.utils.data_io import write_results_to_uea_format
from sktime.utils.sampling import stratified_resample


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


def run_classification_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier,
    results_path,
    cls_name="",
    dataset="",
    resample_id=0,
    train_file=False,
    test_file=True,
):
    """Run a classification experiment and save the results to file.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    X_train : pd.DataFrame or np.array
        The data to train the classifier.
    y_train : np.array, default = None
        Training data class labels.
    X_test : pd.DataFrame or np.array, default = None
        The data used to test the trained classifier.
    y_test : np.array, default = None
        Testing data class labels.
    classifier : BaseClassifier
        Classifier to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    cls_name : str, default=""
        Name of the classifier.
    dataset : str, default=""
        Name of problem.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    test_file : bool, default=True:
         Whether to generate test files or not. If the classifier can generate its own
         train probabilities, the classifier will be built but no file will be output.
    """
    if not test_file and not train_file:
        raise Exception(
            "Both test_file and train_file are set to False. "
            "At least one must be output."
        )

    classifier_train_probs = (
        train_file and getattr(classifier, "time_limit_in_minutes", None) is not None
    )
    build_time = -1

    if test_file or classifier_train_probs:
        start = int(round(time.time() * 1000))
        classifier.fit(X_train, y_train)
        build_time = int(round(time.time() * 1000)) - start

        if test_file:
            start = int(round(time.time() * 1000))
            probs = classifier.predict_proba(X_test)
            test_time = int(round(time.time() * 1000)) - start

            if "composite" in cls_name.lower():
                second = "Para info too long!"
            else:
                second = str(classifier.get_params())
            second.replace("\n", " ")
            second.replace("\r", " ")

            preds = classifier.classes_[np.argmax(probs, axis=1)]
            acc = accuracy_score(y_test, preds)
            third = (
                str(acc)
                + ","
                + str(build_time)
                + ","
                + str(test_time)
                + ",-1,-1,"
                + str(len(classifier.classes_))
                + ",,-1,-1"
            )

            write_results_to_uea_format(
                second_line=second,
                third_line=third,
                first_line_comment="PREDICTIONS,Generated by experiments.py on "
                + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                + ".",
                timing_type="MILLISECONDS",
                output_path=results_path,
                estimator_name=cls_name,
                resample_seed=resample_id,
                y_pred=preds,
                predicted_probs=probs,
                dataset_name=dataset,
                y_true=y_test,
                split="TEST",
                full_path=False,
            )

    if train_file:
        start = int(round(time.time() * 1000))
        if classifier_train_probs:  # Normally Can only do this if test has been built
            train_probs = classifier._get_train_probs(X_train, y_train)
        else:
            cv_size = 10
            _, counts = np.unique(y_train, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class

            train_probs = cross_val_predict(
                classifier, X=X_train, y=y_train, cv=cv_size, method="predict_proba"
            )
        train_time = int(round(time.time() * 1000)) - start

        if "composite" in cls_name.lower():
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        train_preds = classifier.classes_[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(y_train, train_preds)
        third = (
            str(train_acc)
            + ","
            + str(build_time)
            + ",-1,-1,-1,"
            + str(len(classifier.classes_))
            + ",,"
            + str(train_time)
            + ","
            + str(build_time + train_time)
        )

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            first_line_comment="PREDICTIONS,Generated by experiments.py on "
            + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            + ".",
            timing_type="NANOSECONDS",
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=train_preds,
            predicted_probs=train_probs,
            dataset_name=dataset,
            y_true=y_train,
            split="TRAIN",
            full_path=False,
        )


def load_and_run_classification_experiment(
    problem_path,
    results_path,
    cls_name,
    dataset,
    classifier=None,
    resample_id=0,
    overwrite=False,
    build_train=False,
    predefined_resample=False,
):
    """Load a dataset and run a classification experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    cls_name : str
        Determines which classifier to use, as defined in set_classifier. This assumes
        predict_proba is implemented, to avoid predicting twice. May break some
        classifiers though.
    dataset : str
        Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+"_TRAIN.ts",
        same for "_TEST".
    classifier : BaseClassifier, default=None
        Classifier to be used in the experiment, if none is provided one is selected
        using cls_name using resample_id as a seed.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    build_train : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    predefined_resample : bool, default=False
        Read a predefined resample from file instead of performing a resample. If True
        the file format must include the resample_id at the end of the dataset name i.e.
        <problem_path>/<dataset>/<dataset>+<resample_id>+"_TRAIN.ts".
    """
    # Check which files exist, if both exist, exit
    build_test = True
    if not overwrite:
        full_path = (
            results_path
            + "/"
            + cls_name
            + "/Predictions/"
            + dataset
            + "/testResample"
            + str(resample_id)
            + ".csv"
        )

        if os.path.exists(full_path):
            build_test = False

        if build_train:
            full_path = (
                results_path
                + "/"
                + cls_name
                + "/Predictions/"
                + dataset
                + "/trainResample"
                + str(resample_id)
                + ".csv"
            )

            if os.path.exists(full_path):
                build_train = False

        if build_train is False and build_test is False:
            return

    if predefined_resample:
        X_train, y_train = load_ts(
            problem_path + dataset + "/" + dataset + str(resample_id) + "_TRAIN.ts"
        )
        X_test, y_test = load_ts(
            problem_path + dataset + "/" + dataset + str(resample_id) + "_TEST.ts"
        )
    else:
        X_train, y_train = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN.ts")
        X_test, y_test = load_ts(problem_path + dataset + "/" + dataset + "_TEST.ts")
        if resample_id != 0:
            X_train, y_train, X_test, y_test = stratified_resample(
                X_train, y_train, X_test, y_test, resample_id
            )

    if classifier is None:
        classifier = set_classifier(cls_name, resample_id, build_train)

    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path,
        cls_name=cls_name,
        dataset=dataset,
        resample_id=resample_id,
        train_file=build_train,
        test_file=build_test,
    )


def set_classifier(cls, resample_id=None, train_file=False):
    """Construct a classifier.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.

    Parameters
    ----------
    cls : str
        String indicating which classifier you want.
    resample_id : int or None, default=None
        Classifier random seed.
    train_file : bool, default=False
        Whether a train file is being produced.

    Return
    ------
    classifier : A BaseClassifier.
        The classifier matching the input classifier name.
    """
    name = cls.lower()
    # Dictionary based
    if name == "boss" or name == "bossensemble":
        return BOSSEnsemble(random_state=resample_id)
    elif name == "cboss" or name == "contractableboss":
        return ContractableBOSS(random_state=resample_id)
    elif name == "tde" or name == "temporaldictionaryensemble":
        return TemporalDictionaryEnsemble(
            random_state=resample_id, save_train_predictions=train_file
        )
    elif name == "weasel":
        return WEASEL(random_state=resample_id)
    elif name == "muse":
        return MUSE(random_state=resample_id)
    # Distance based
    elif name == "pf" or name == "proximityforest":
        return ProximityForest(random_state=resample_id)
    elif name == "pt" or name == "proximitytree":
        return ProximityTree(random_state=resample_id)
    elif name == "ps" or name == "proximityStump":
        return ProximityStump(random_state=resample_id)
    elif name == "dtwcv" or name == "kneighborstimeseriesclassifier":
        return KNeighborsTimeSeriesClassifier(distance="dtwcv")
    elif name == "dtw" or name == "1nn-dtw":
        return KNeighborsTimeSeriesClassifier(distance="dtw")
    elif name == "msm" or name == "1nn-msm":
        return KNeighborsTimeSeriesClassifier(distance="msm")
    elif name == "ee" or name == "elasticensemble":
        return ElasticEnsemble(random_state=resample_id)
    elif name == "shapedtw":
        return ShapeDTW()
    # Feature based
    elif name == "catch22":
        return Catch22Classifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "matrixprofile":
        return MatrixProfileClassifier(random_state=resample_id)
    elif name == "signature":
        return SignatureClassifier(
            random_state=resample_id,
            classifier=RandomForestClassifier(n_estimators=500),
        )
    elif name == "tsfresh":
        return TSFreshClassifier(
            random_state=resample_id, estimator=RandomForestClassifier(n_estimators=500)
        )
    elif name == "tsfresh-r":
        return TSFreshClassifier(
            random_state=resample_id,
            estimator=RandomForestClassifier(n_estimators=500),
            relevant_feature_extractor=True,
        )
    # Hybrid
    elif name == "hc1" or name == "hivecotev1":
        return HIVECOTEV1(random_state=resample_id)
    elif name == "hc2" or name == "hivecotev2":
        return HIVECOTEV2(random_state=resample_id)
    # Interval based
    elif name == "rise" or name == "randomintervalspectralforest":
        return RandomIntervalSpectralForest(random_state=resample_id, n_estimators=500)
    elif name == "tsf" or name == "timeseriesforestclassifier":
        return TimeSeriesForestClassifier(random_state=resample_id, n_estimators=500)
    elif name == "cif" or name == "canonicalintervalforest":
        return CanonicalIntervalForest(random_state=resample_id, n_estimators=500)
    elif name == "stsf" or name == "supervisedtimeseriesforest":
        return SupervisedTimeSeriesForest(random_state=resample_id, n_estimators=500)
    elif name == "drcif":
        return DrCIF(
            random_state=resample_id, n_estimators=500, save_transformed_data=train_file
        )
    # Kernel based
    elif name == "rocket":
        return ROCKETClassifier(random_state=resample_id)
    elif name == "arsenal":
        return Arsenal(random_state=resample_id, save_transformed_data=train_file)
    # Shapelet based
    elif name == "stc" or name == "shapelettransformclassifier":
        return ShapeletTransformClassifier(
            random_state=resample_id, save_transformed_data=train_file
        )
    elif name == "mrseql" or name == "mrseqlclassifier":
        return MrSEQLClassifier(seql_mode="fs", symrep=["sax", "sfa"])
    else:
        raise Exception("UNKNOWN CLASSIFIER")
