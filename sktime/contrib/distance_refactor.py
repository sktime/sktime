# -*- coding: utf-8 -*-
"""Test the move from (m,d) to (d,m)."""
import numpy as np

from sktime.datasets import load_basic_motions, load_unit_test


def difference_test():
    """Test the distance functions with allowable input."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    d1 = X_train[0]
    d2 = X_train[1]
    d3 = np.array([1, 2, 3, 4, 5, 6])
    d4 = np.array([3, 4, 5, 6, 7, 8])

    d = ddtw_distance(d1, d2)
    print(" distance = ", d)
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    c1 = X_train[0]
    c2 = X_train[1]
    c3 = np.array([[1, 2, 3, 4, 5, 6]])
    c4 = np.array([[3, 4, 5, 6, 7, 8]])
    d = ddtw_distance(c1, c2)
    print(" distance = ", d)
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    b1 = X_train[0]
    b2 = X_train[1]
    b3 = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    b4 = np.array([[3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6]])
    d = ddtw_distance(b1, b2)
