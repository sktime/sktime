# -*- coding: utf-8 -*-
"""Test the move from (m,d) to (d,m)."""
import numpy as np

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
    print("Shape  = ", d1.shape)
    name = "DTW"
    dist1 = dtw_distance(d1, d2, window=0.0)
    print(name, " w = 0 dist = ", dist1)
    dist1 = dtw_distance(d1, d2, window=0.1)
    print(name, " w = 0.1. dist 1 = ", dist1)
    dist1 = dtw_distance(d1, d2, window=1.0)
    print(name, " w = 1 dist 1 = ", dist1)
    print(" SHAPE  = ", d1.shape)
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    print(" SHAPE  = ", X_train[0].shape)
    name = "DTW"
    dist1 = dtw_distance(d1, d2, window=0.0)
    print(name, " g = 0 dist = ", dist1)
    dist1 = dtw_distance(d1, d2, window=0.1)
    print(name, " g = 0.1. dist 1 = ", dist1)
    dist1 = wddtw_distance(d1, d2, window=1.0)
    print(name, " g window = 1 dist 1 = ", dist1)
    print(" SHAPE  = ", d1.shape)
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    print(" SHAPE  = ", X_train[0].shape)

    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    b1 = X_train[0]
    b2 = X_train[1]
    #    b1 = np.transpose(b1)
    #    b2 = np.transpose(b2)
    print("BM shape = ", b1.shape)
    dist2 = dtw_distance(b1, b2, window=0.0)
    print(" g = 0.0, BASIC MOTIONS DIST = ", dist2)
    dist2 = dtw_distance(b1, b2, window=0.1)
    print(" g = 0.1, BASIC MOTIONS DIST = ", dist2)
    dist2 = dtw_distance(b1, b2, window=1.0)
    print(" g = 1, BASIC MOTIONS DIST = ", dist2)

    dist2 = euclidean_distance(b1, b2)
    print(" ED BASIC MOTIONS DIST = ", dist2)


#  print(" Window = 1, BASIC MOTIONS DIST = ", dist2)


if __name__ == "__main__":
    difference_test()
