# -*- coding: utf-8 -*-
"""Test the move from (m,d) to (d,m)."""
import numpy as np
from sklearn import metrics

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.distances import (
    ddtw_distance,
    dtw_distance,
    edr_distance,
    erp_distance,
    euclidean_distance,
    lcss_distance,
    msm_distance,
    wddtw_distance,
    wdtw_distance,
)

# Clustering Wtih num custers set to 2 and transpose
expected_rand_unit_test = {
    "euclidean": 0.5210526315789473,
    "dtw": 0.5210526315789473,
    "erp": 0.47368421052631576,
    "edr": 0.4789473684210526,
    "wdtw": 0.5210526315789473,
    "lcss": 0.7315789473684211,
    "msm": 0.6052631578947368,
    "ddtw": 0.6631578947368421,
    "wddtw": 0.6052631578947368,
}
expected_rand_basic_motions = {
    "euclidean": 0.5947368421052631,
    "dtw": 0.5421052631578948,
    "erp": 0.47368421052631576,
    "edr": 0.4842105263157895,
    "wdtw": 0.5947368421052631,
    "lcss": 0.7,
    "msm": 0.7947368421052632,
    "ddtw": 0.5736842105263158,
    "wddtw": 0.6684210526315789,
}

distances = [
    "euclidean",
    "dtw",
    "wdtw",
    #    "erp",
    #    "edr",
    #    "lcss",
    #    "msm",
    #    "ddtw",
    #    "wddtw",
]


def debug_clusterers():
    """Debug clusterers."""
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    #    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    #   X_train2, y_train2 = load_unit_test(split="train", return_type="numpy2d")
    parameters = {"window": 1.0, "epsilon": 50.0, "g": 0.05, "c": 1.0}
    for dist in distances:
        kmeans = TimeSeriesKMeans(
            averaging_method="mean",
            random_state=1,
            n_init=2,
            n_clusters=2,
            init_algorithm="kmeans++",
            metric=dist,
            distance_params=parameters,
        )
        kmeans.fit(X_train)
        y_pred = kmeans.predict(X_train)
        train_rand = metrics.rand_score(y_train, y_pred)
        print('"' + dist + '": ' + str(train_rand) + ",")
        # kmeans.fit(X_train2)
        # y_pred = kmeans.predict(X_train2)
        # train_rand2 = metrics.rand_score(y_train2, y_pred)
        # assert train_rand == train_rand2
        # print(" Rand score on with 2D unit test = ",train_rand)


def difference_test():
    """Test the distance functions with allowable input.

    TEST 1: Distances. Generate all distances with tsml, compare.
    TEST 2: Classification.
    TEST 3: Clustering.
    TEST 4: tslearn
    """
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    d1 = X_train[0]
    d2 = X_train[2]
    dist = msm_distance
    #    d1=np.transpose(d1)
    #    d2=np.transpose(d2)
    print("Shape  = ", d1.shape)
    name = "msm"
    no_window = np.zeros((d1.shape[1], d2.shape[1]))
    dist1 = dist(d1, d2, c=0.0)
    print(name, " w = 0 dist = ", dist1)
    dist1 = dist(d1, d2, c=0.1)
    print(name, " w = 0.1. dist 1 = ", dist1)
    dist1 = dist(d1, d2, c=1)
    print(name, " w = 1 dist 1 = ", dist1)
    print(" SHAPE  = ", d1.shape)
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    b1 = X_train[0]
    b2 = X_train[1]
    #    b1 = np.transpose(b1)
    #    b2 = np.transpose(b2)
    dist = dtw_distance
    print("BM shape = ", b1.shape)
    dist2 = dist(b1, b2, epsilon=0.0)
    print(" g = 0.0, BASIC MOTIONS DIST = ", dist2)
    dist2 = dist(b1, b2, epsilon=1.0)
    print(" g = 0.1, BASIC MOTIONS DIST = ", dist2)
    dist2 = dist(b1, b2, epsilon=4.0)
    print(" g = 1, BASIC MOTIONS DIST = ", dist2)


#   dist2 = euclidean_distance(b1, b2)
#   print(" ED BASIC MOTIONS DIST = ", dist2)


#  print(" Window = 1, BASIC MOTIONS DIST = ", dist2)


if __name__ == "__main__":
    #    debug_clusterers()
    difference_test()
