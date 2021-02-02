# This will become tests, but its a work in progress
import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from classification.shapelet_based import ROCKETClassifier
from sktime.classification.base import classifier_list
from sktime.contrib.experiments import set_classifier
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.utils.validation.panel import check_X
from sktime.utils.data_processing import from_2d_array_to_nested
import sktime.datasets.base as sktime
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_processing import convert_from_dictionary
from sktime.distances.elastic import euclidean_distance
from sktime.distances.elastic_cython import ddtw_distance
from sktime.distances.elastic_cython import dtw_distance
from sktime.distances.elastic_cython import erp_distance
from sktime.distances.elastic_cython import lcss_distance
from sktime.distances.elastic_cython import msm_distance
from sktime.distances.elastic_cython import twe_distance
from sktime.distances.elastic_cython import wddtw_distance
from sktime.distances.elastic_cython import wdtw_distance

from sktime.classification.dictionary_based import (
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
    WEASEL,
    MUSE,
)

# NOTES to check: RISE and TSF do not exit well if series < min interval
# Cannot get KNearestTimeSeriesNeighbors to work properly.
classifier_list = [
    # in classification/distance_based
    "ProximityForest",
    "KNeighborsTimeSeriesClassifier",
    "ElasticEnsemble",
    "ShapeDTW",
    # in classification/dictionary_based
    "BOSS",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "WEASEL",
    "MUSE",
    # in classification/interval_based
    "RandomIntervalSpectralForest",
    "TimeSeriesForest",
    "CanonicalIntervalForest",
    # in classification/shapelet_based
    "ShapeletTransformClassifier",
    "ROCKET",
    "MrSEQLClassifier",
]

distance_functions = ["euclidean","dtw","dtwcv","ddtw","wdtw","wddtw","lcss","erp",
                      "msm"]


# ISSUES
# 1. Euclidean distance not working
# 2. Difference classifiers all predicting zero
# 3. Possible error in transposition
# TODO
# Focus on DTW full window only.
# See if we can print distances
#
#
#        printf(f" Dimensions of the differences = {X[0].shape} and {Y[0].shape}")

def debug_distances():
    x1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]])
    x2 = np.array([[6.0, 11.0, 15.0, 2.0, 7.0, 1.0]])
#    x1 = np.array([[1.0, 1.0, 3.0, 1.0, 1.0, 1.0]])
#    x2 = np.array([[1.0, 1.0, 1.0, 1.0, 3.0, 1.0]])

    x1 = x1.transpose()
    x2 = x2.transpose()

#    print(f" Shape of x 1 = {x1.shape}  values = \n{x1}")
#    print(f" Shape of x 2 = {x2.shape}  values = \n{x2}")

    d = euclidean_distance(x1, x2)
    print(f" Euclidean distance = {d}")
    d = dtw_distance(x1, x2, w=1)
    print(f" DTW distance = {d}")
    d = wdtw_distance(x1, x2)
    print(f" WDTW distance = {d}")
    d = lcss_distance(x1, x2)
    print(f"LCSS distance = {d}")
    d = erp_distance(x1, x2)
    print(f" ERP distance = {d}")
    d = msm_distance(x1, x2)
    print(f" MSM distance = {d}")

def debug_knn():
    path = os.path.join(sktime.MODULE, "data")
    problem = "Gunpoint"
    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join(path, f"{problem}/{problem}_TRAIN.ts")
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        os.path.join(path, f"{problem}/{problem}_TEST.ts")
    )
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)
#    metric = dtw_distance
#    d = dtw_distance(X[0], X[1])
#    print(f" distance ={d}")
    for m in distance_functions:
        print(f" Building {m} ....")
        classifier = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric=m)
#        classifier = KNeighborsTimeSeriesClassifier(metric=m, algorithm="ball_tree")
#        print(f" All tags for knn = {classifier._get_tags()}")
#        print(f" Tags for knn = {classifier._more_tags()}")
        classifier.fit(train_x, train_y)
        print(f" Fit {m} complete, predicting on itself...")
        preds = classifier.predict(test_x)
        ac = accuracy_score(test_y, preds)
        print(f" Testing KNN with {m} ac = {ac} preds = {preds} actual  = {train_y}")


def check_file_loaded():
    """
    Check a classifier can take as input either an nested panda or numpy array
    Valid input
    """
    for i in range(0, len(classifier_list)):
        path = os.path.join(sktime.MODULE, "data")
        train_x, train_y = load_from_tsfile_to_dataframe(
            os.path.join(path, "UnitTest/UnitTest_TRAIN.ts")
        )
        test_x, test_y = load_from_tsfile_to_dataframe(
           os.path.join(path, "UnitTest/UnitTest_TEST.ts")
        )

        classifier = set_classifier(classifier_list[i])
        classifier.fit(train_x, train_y)
        preds = classifier.predict(test_x)
        ac = accuracy_score(test_y, preds)
        print(f" Testing {classifier_list[i]} ac = {ac}")


def check_multivariate_pandas():
    """
    Check a classifier can take as input either an nested panda or numpy array
    Valid input
    """
    for i in range(0,1):
        x = {
            "Series1": [ [1.0,2.0,3.0,1.0,2.0],[1.0,2.0,3.0,1.0,2.0] ],
            "Series2": [ [1.0,2.0,3.0,1.0,2.0],[1.0,2.0,3.0,1.0,2.0] ],
            "Series3": [ [11.0, 12.0, 13.0, 11.0, 12.0], [111.0, 112.0, 113.0, 111.0,
                                                          112.0]],
        }
        panda = pd.DataFrame(x)
        print(panda)
        panda = panda.transpose()
        print(panda)


def check_dictionary():
    """
    build a classifier from a dictionary of series
    """
    x_train = {
        "Series1": [1.0, 2.0, 3.0, 1.0, 2.0],
        "Series2": [3.0, 2.0, 1.0, 3.0, 2.0],
        "Series3": [1.0, 2.0, 3.0, 1.0, 2.0],
        "Series4": [3.0, 2.0, 1.0, 3.0, 2.0],
    }
    x_train = convert_from_dictionary(x_train)
    y = np.array([0, 1, 1, 1], np.int32)
    classifier = ROCKETClassifier()
    classifier.fit(x_train, y)

    print(" Build Finished ")
    x_test = {
        "Series5": [1.0, 2.0, 3.0, 1.0, 2.0],
        "Series6": [1.0, 2.0, 3.0, 4.0, 1.0]
    }
    x_test = convert_from_dictionary(x_test)
    y_pred = classifier.predict(x_test)
    print(f" classifier ROCKET preds = {y_pred}")


def check_pandas():
    """
    Check a classifier can take as input either an nested panda or numpy array
    Valid input
    """
    for i in range(0, len(classifier_list)):
        x = {
            "Series1": [1.0,2.0,3.0,1.0,2.0],
            "Series2": [3.0,2.0,1.0,3.0,2.0],
            "Series3": [1.0, 2.0, 3.0, 1.0, 2.0],
            "Series4": [3.0, 2.0, 1.0, 3.0, 2.0],
            "Series5": [1.0, 2.0, 3.0, 1.0, 2.0],
            "Series6": [3.0, 2.0, 1.0, 3.0, 2.0],
            "Series7": [1.0, 2.0, 3.0, 1.0, 2.0],
            "Series8": [3.0, 2.0, 1.0, 3.0, 2.0],
        }
        X = convert_from_dictionary(x)

        #panda = pd.DataFrame(x)
        #panda = panda.transpose()
        #print(panda)
        #X = from_2d_array_to_nested(panda)
        y = np.array([0, 1,1,1,0,0,1,1], np.int32)
        classifier = set_classifier(classifier_list[i])
        classifier.fit(X, y)
        print(" Build Finished ")
        x_n = {
            "Series7": [1.0, 2.0, 3.0, 1.0, 2.0],
            "Series8": [1.0, 2.0, 3.0, 1.0, 2.0]
        }
        x_new = pd.DataFrame(x_n)
        x_new = x_new.transpose()
        X = from_2d_array_to_nested(x_new)
        y_pred = classifier.predict(X)
        print(f" classifier {classifier_list[i]} preds = {y_pred}")

 #   y_prob = classifier.predict_proba(x_new)
  #  print(y_prob)


def check_npyarray():
    """
    Check a classifier can take as input either an nested panda or numpy array
    Valid input
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]], np.int32)
    y = np.array([0, 1, 0], np.int32)
    X = from_2d_array_to_nested(x)
    classifier = TimeSeriesForest()
    classifier.fit(X, y)
    x_new = np.array([1, 1, 3])
    y_pred = classifier.predict(x_new)


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """
    # check_dictionary();
    # check_pandas()
    # check_multivariate_pandas()
    # check_file_loaded()
    #debug_knn()
    debug_distances()

