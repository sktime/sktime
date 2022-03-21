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
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    d1 = X_train[0]
    d2 = X_train[2]
    d3 = np.array([1, 2, 3, 4, 5, 6])
    d4 = np.array([3, 4, 5, 6, 7, 8])
    #    d1 = d1.transpose()
    #    d1 = d2.transpose()
    print("Shape  = ", d1.shape)
    name = "DDTW"
    dist1 = euclidean_distance(d1, d2, window=0.0)
    print(name, " w = 0 dist = ", dist1)
    dist1 = euclidean_distance(d1, d2, window=0.1)
    print(name, " w = 0.1. dist 1 = ", dist1)
    dist1 = euclidean_distance(d1, d2, window=1.0)
    print(name, " w = 1 dist 1 = ", dist1)
    print(" SHAPE  = ", d1.shape)
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    print(" SHAPE  = ", X_train[0].shape)
    name = "WDDTW"
    dist1 = wddtw_distance(d1, d2, g=0.0)
    print(name, " g = 0 dist = ", dist1)
    dist1 = wddtw_distance(d1, d2, g=0.1)
    print(name, " g = 0.1. dist 1 = ", dist1)
    dist1 = wddtw_distance(d1, d2, g=1.0)
    print(name, " g window = 1 dist 1 = ", dist1)
    print(" SHAPE  = ", d1.shape)
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    print(" SHAPE  = ", X_train[0].shape)

    c1 = X_train[0]
    c2 = X_train[1]
    c3 = np.array([[1, 2, 3, 4, 5, 6]])
    c4 = np.array([[3, 4, 5, 6, 7, 8]])
    dist1 = dtw_distance(c1, c2)
    dist2 = dtw_distance(c3, c4)
    #    print(" SHAPE  = ", c1.shape)
    #    print(" distance (1,m) input d1 = ", dist1, " distance 2 = ", dist2)
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    b1 = X_train[0]
    b2 = X_train[1]
    b3 = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    b4 = np.array([[3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6]])
    dist1 = dtw_distance(b1, b2)
    dist2 = dtw_distance(b3, b4)
    #   print(" 3D load SHAPE  = ", b1.shape)
    #    print(" distance (d,m) input d1 = ", dist1, " distance 2 = ", dist2)
    a1 = c1.transpose()
    a2 = c2.transpose(1, 0)
    a3 = b1.transpose()
    a4 = b1.transpose()
    #    print(" Wrong shape univariate = ", a1.shape)
    #    print(" Wrong shape multi = ", a3.shape)
    dist1 = dtw_distance(a1, a2)
    #    print(" distance 1 = ", dist1)
    dist2 = dtw_distance(a3, a4)
    #   print(" distance 2 = ", dist2)
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    dist2 = dtw_distance(X_train[0], X_train[1], window=0.0)
    #   print(" Window = 0.0, BASIC MOTIONS DIST = ", dist2)
    dist2 = dtw_distance(X_train[0], X_train[1], window=0.1)
    #   print(" Window = 0.1, BASIC MOTIONS DIST = ", dist2)
    dist2 = dtw_distance(X_train[0], X_train[1], window=1.0)


#  print(" Window = 1, BASIC MOTIONS DIST = ", dist2)


if __name__ == "__main__":
    difference_test()
