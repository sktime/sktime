# -*- coding: utf-8 -*-
""" experiments.py: method to run experiments as an alternative to orchestration.

Note ProximityForest and ElasticEnsemble use cython implementations of the distance measures. You need a c++ compiler for this.
These are notes mainly for me.
On  windows:
The easiest way to install visual studio. Then, from an anaconda prompt, change to the sktime dir, then
python setup.py install
python setup.py build_ext -i
(may not be necessary with pipinstall).

on the cluster
copy source over then as above,
enter interactive mode, got to sktime root
IF not done before,

1) >interactive
2) change dir to sktime
3) module add python/anaconda/2019.3/3.7
4) conda init bash
5) conda create -n sktime
6) conda activate sktime
7) conda install pip
8) pip install setuptools scipy cython numpy pandas scikit-learn pytest statsmodels
9) export PYTHONPATH=$(pwd)
10) python <FULLPATH>setup.py install
11) python <FULLPATH>setup.py build_ext -i

then run sktime.sh script

NOTE: do
"""

import os

import sklearn.preprocessing
import sklearn.utils

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
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.stattools import acf

import sktime.classification.compose._ensemble as ensemble
import sktime.classification.dictionary_based._boss as db
import sktime.classification.interval_based._rise as fb
import sktime.classification.interval_based._tsf as ib
import sktime.classification.distance_based._elastic_ensemble as dist
import sktime.classification.distance_based._time_series_neighbors as nn
import sktime.classification.distance_based._proximity_forest as pf
import sktime.classification.shapelet_based._stc as st
from sktime.classification.distance_based._shape_dtw import ShapeDTW
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts
from sktime.transformations.panel.compose import make_row_transformer
from sktime.transformations.panel.segment import RandomIntervalSegmenter

from sktime.transformations.panel.reduce import Tabularizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import sktime.datasets.tsc_dataset_names as dataset_lists

__author__ = "Anthony Bagnall"

""" Prototype mechanism for testing classifiers on the UCR format. This mirrors the mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method recommended here, they can be directly
and automatically compared to the results generated in java

Will have both low level version and high level orchestration version soon.
"""


# Used on lines 410 and 411
def _normalise_X(X):
    """Helper function to normalise X using the z-score standardisation"""
    # get the number of attributes and instances
    num_atts = X.shape[1]
    num_insts = X.shape[0]

    df = pd.DataFrame()

    for x in X.columns:
        col = X[x]
        data = []
        for y in range(num_insts):
            ser = col.iloc[y]
            ser = ser.to_numpy()
            sd = np.std(ser)
            avg = np.mean(ser)
            ser = (ser - avg) / sd
            data.append(pd.Series(ser))
        df[x] = data

    return df


def set_classifier(cls, resampleId):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """
    if cls.lower() == "pf":
        return pf.ProximityForest(random_state=resampleId)
    elif cls.lower() == "pt":
        return pf.ProximityTree(random_state=resampleId)
    elif cls.lower() == "ps":
        return pf.ProximityStump(random_state=resampleId)
    elif cls.lower() == "rise":
        return fb.RandomIntervalSpectralForest(random_state=resampleId)
    elif cls.lower() == "tsf":
        return ib.TimeSeriesForest(random_state=resampleId)
    elif cls.lower() == "boss":
        return db.BOSSEnsemble()
    elif cls.lower() == "st":
        return st.ShapeletTransformClassifier(time_contract_in_mins=1500)
    elif cls.lower() == "dtw":
        return nn.KNeighborsTimeSeriesClassifier(distance="dtw")
    elif cls.lower() == "ee" or cls.lower() == "elasticensemble":
        return dist.ElasticEnsemble()
    elif cls.lower() == "shapedtw_raw":
        return ShapeDTW(
            subsequence_length=30, shape_descriptor_function="raw", metric_params=None
        )
    elif cls.lower() == "shapedtw_dwt":
        return ShapeDTW(
            subsequence_length=30,
            shape_descriptor_function="dwt",
            metric_params={"num_levels_dwt": 3},
        )
    elif cls.lower() == "shapedtw_paa":
        return ShapeDTW(
            subsequence_length=30,
            shape_descriptor_function="paa",
            metric_params={"num_intervals_paa": 5},
        )
    elif cls.lower() == "shapedtw_slope":
        return ShapeDTW(
            subsequence_length=30,
            shape_descriptor_function="slope",
            metric_params={"num_intervals_slope": 5},
        )
    elif cls.lower() == "shapedtw_hog1d":
        return ShapeDTW(
            subsequence_length=30,
            shape_descriptor_function="hog1d",
            metric_params={
                "num_bins_hog1d": 8,
                "num_intervals_hog1d": 2,
                "scaling_factor_hog1d": 0.1,
            },
        )
    elif cls.lower() == "tsfcomposite":
        # It defaults to TSF
        return ensemble.ComposableTimeSeriesForestClassifier()
    elif cls.lower() == "risecomposite":
        steps = [
            ("segment", RandomIntervalSegmenter(n_intervals=1, min_length=5)),
            (
                "transform",
                FeatureUnion(
                    [
                        (
                            "acf",
                            make_row_transformer(
                                FunctionTransformer(func=acf_coefs, validate=False)
                            ),
                        ),
                        (
                            "ps",
                            make_row_transformer(
                                FunctionTransformer(func=powerspectrum, validate=False)
                            ),
                        ),
                    ]
                ),
            ),
            ("tabularise", Tabularizer()),
            ("clf", DecisionTreeClassifier()),
        ]
        base_estimator = Pipeline(steps)
        return ensemble.ComposableTimeSeriesForestClassifier(
            estimator=base_estimator, n_estimators=100
        )
    else:
        raise Exception("UNKNOWN CLASSIFIER")


def acf_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    return acf(x, nlags=nlags).ravel()


def powerspectrum(x, **kwargs):
    x = np.asarray(x).ravel()
    fft = np.fft.fft(x)
    ps = fft.real * fft.real + fft.imag * fft.imag
    return ps[: ps.shape[0] // 2].ravel()


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
    Method to run a basic experiment and write the results to files called testFold<resampleID>.csv and, if required,
    trainFold<resampleID>.csv.
    :param problem_path: Location of problem files, full path.
    :param results_path: Location of where to write results. Any required directories will be created
    :param cls_name: determines which classifier to use, as defined in set_classifier. This assumes predict_proba is
    implemented, to avoid predicting twice. May break some classifiers though
    :param dataset: Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+"_TRAIN"+format, same for "_TEST"
    :param resampleID: Seed for resampling. If set to 0, the default train/test split from file is used. Also used in output file name.
    :param overwrite: if set to False, this will only build results if there is not a result file already present. If
    True, it will overwrite anything already there
    :param format: Valid formats are ".ts", ".arff" and ".long". For more info on format, see
    https://github.com/alan-turing-institute/sktime/blob/master/examples/Loading%20Data%20Examples.ipynb
    :param train_file: whether to generate train files or not. If true, it performs a 10xCV on the train and saves
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

    # TO DO: Automatically differentiate between problem types, currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" + format)
    testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)

    trainX = _normalise_X(trainX)
    testX = _normalise_X(testX)

    if resampleID != 0:
        # allLabels = np.concatenate((trainY, testY), axis = None)
        # allData = pd.concat([trainX, testX])
        # train_size = len(trainY) / (len(trainY) + len(testY))
        # trainX, testX, trainY, testY = train_test_split(allData, allLabels, train_size=train_size,
        #                                                                random_state=resampleID, shuffle=True,
        #                                                                stratify=allLabels)
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
        ):  # Normally Can only do this if test has been built ... well not necessarily true, but will do for now
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
    This is very alpha and I will probably completely change the structure once train fold is sorted, as that internally
    does all this I think!
    Output mirrors that produced by this Java
    https://github.com/TonyBagnall/uea-tsc/blob/master/src/main/java/experiments/Experiments.java
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
            "The number of predicted class values is not the same as the number of actual class values"
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

    # print(classifier_name+" on "+dataset_name+" for resample "+str(resample_seed)+"   "+train_or_test+" data has line three "+third_line)
    # the first line of the output file is in the form of:
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

    # the second line of the output is free form and classifier-specific; usually this will record info
    # such as parameter options used, any constituent model names for ensembles, etc.
    file.write(str(second_line) + "\n")

    # the third line of the file is the accuracy (should be between 0 and 1 inclusive). If this is a train
    # output file then it will be a training estimate of the classifier on the training data only (e.g.
    # 10-fold cv, leave-one-out cv, etc.). If this is a test output file, it should be the output
    # of the estimator on the test data (likely trained on the training data for a-priori parameter optimisation)
    file.write(str(third_line))
    file.write("\n")

    # from line 4 onwards each line should include the actual and predicted class labels (comma-separated). If
    # present, for each case, the probabilities of predicting every class value for this case should also be
    # appended to the line (a space is also included between the predicted value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   actual_class_val[i], predicted_class_val[i],,prob_class_0[i],prob_class_1[i],...,prob_class_c[i]
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


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """

    #    test_loading()
    #    sys.exit()
    #    print('experimenting...')
    # Input args -dp=${dataDir} -rp=${resultsDir} -cn=${classifier} -dn=${dataset} -f=\$LSB_JOBINDEX
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
        #        data_dir = "/scratch/univariate_datasets/"
        #        results_dir = "/scratch/results"
        data_dir = "/bench/datasets/Univariate2018/"
        results_dir = "C:/Users/ajb/Dropbox/Turing Project/Results/"
        # data_dir = "Z:/ArchiveData/Univariate_ts/"
        # results_dir = "E:/Temp/"
        #        results_dir = "Z:/Results/sktime Bakeoff/"
        dataset = "ItalyPowerDemand"
        trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
        testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
        classifier = "TSF"
        resample = 1
        #         for i in range(0, len(univariate_datasets)):
        #             dataset = univariate_datasets[i]
        # #            print(i)
        # #            print(" problem = "+dataset)
        tf = False
        run_experiment(
            overwrite=True,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=classifier,
            dataset=dataset,
            resampleID=resample,
            train_file=tf,
        )
