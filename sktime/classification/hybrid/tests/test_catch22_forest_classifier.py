# -*- coding: utf-8 -*-
import sys

import numpy as np
import pytest
from numpy import testing

from sktime.classification.hybrid import Catch22ForestClassifier
from sktime.datasets import load_gunpoint, load_basic_motions


@pytest.mark.skipif(sys.platform == 'win32',
                    reason="Not supported for Windows currently.")
def test_catch22_forest_classifier_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train c22f
    c22f = Catch22ForestClassifier(random_state=0)
    c22f.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22f.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas,
                               catch22_forest_classifier_gunpoint_probas)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason="Not supported for Windows currently.")
def test_catch22_forest_classifier_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split='train', return_X_y=True)
    X_test, y_test = load_basic_motions(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train c22f
    c22f = Catch22ForestClassifier(random_state=0)
    c22f.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22f.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas,
                               catch22_forest_classifier_basic_motions_probas)


catch22_forest_classifier_gunpoint_probas = np.array([
    [0.13, 0.87, ],
    [0.26, 0.74, ],
    [0.52, 0.48, ],
    [0.22, 0.78, ],
    [0.16, 0.84, ],
    [0.67, 0.33, ],
    [0.07, 0.93, ],
    [0.46, 0.54, ],
    [0.31, 0.69, ],
    [0.09, 0.91, ],
])
catch22_forest_classifier_basic_motions_probas = np.array([
    [0.97, 0.03, ],
    [0.34, 0.66, ],
    [0.62, 0.38, ],
    [0.03, 0.97, ],
    [0.97, 0.03, ],
    [0.99, 0.01, ],
    [0.35, 0.65, ],
    [0.96, 0.04, ],
    [0.18, 0.82, ],
    [0.2, 0.8, ],
    [0.22, 0.78, ],
    [0.8, 0.2, ],
    [0.0, 1.0, ],
    [0.24, 0.76, ],
    [0.62, 0.38, ],
    [0.96, 0.04, ],
    [0.24, 0.76, ],
    [0.2, 0.8, ],
    [0.64, 0.36, ],
    [0.97, 0.03, ],
])


# def print_array(array):
#     print('[')
#     for sub_array in array:
#         print('[', end='')
#         for value in sub_array:
#             print(value.astype(str), end='')
#             print(', ', end='')
#         print('],')
#     print(']')
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split='train', return_X_y=True)
#     X_test, y_test = load_gunpoint(split='test', return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     c22f_u = Catch22ForestClassifier(random_state=0)
#
#     c22f_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = c22f_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split='train', return_X_y=True)
#     X_test, y_test = load_basic_motions(split='test', return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     c22f_m = Catch22ForestClassifier(random_state=0)
#
#     c22f_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = c22f_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
