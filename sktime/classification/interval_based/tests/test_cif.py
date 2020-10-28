# -*- coding: utf-8 -*-
import sys

import numpy as np
import pytest
from numpy import testing

from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.datasets import load_gunpoint, load_italy_power_demand


@pytest.mark.skipif(
    sys.platform == "win32", reason="Not supported for Windows currently."
)
def test_cif_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=100, random_state=0)
    cif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_gunpoint_probas)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Not supported for Windows currently."
)
def test_cif_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=100, random_state=0)
    cif.fit(X_train, y_train)

    score = cif.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


cif_gunpoint_probas = np.array(
    [
        [
            0.12,
            0.88,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.67,
            0.33,
        ],
        [
            0.48,
            0.52,
        ],
        [
            0.04,
            0.96,
        ],
        [
            0.55,
            0.45,
        ],
        [
            0.25,
            0.75,
        ],
        [
            0.54,
            0.46,
        ],
        [
            0.58,
            0.42,
        ],
        [
            0.1,
            0.9,
        ],
    ]
)


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
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     cif_u = CanonicalIntervalForest(n_estimators=100, random_state=0)
#
#     cif_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = cif_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
