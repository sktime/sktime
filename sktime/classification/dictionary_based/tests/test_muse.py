# -*- coding: utf-8 -*-
"""MUSE test code."""

import numpy as np
from numpy import testing

from sktime.classification.dictionary_based import MUSE
from sktime.datasets import load_basic_motions, load_unit_test


def test_muse_on_unit_test_data():
    """Test of MUSE on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train MUSE
    muse = MUSE(random_state=0, window_inc=4, use_first_order_differences=False)
    muse.fit(X_train, y_train)

    # assert probabilities are the same
    probas = muse.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, muse_unit_test_probas, decimal=2)


def test_muse_on_basic_motions():
    """Test MUSE on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train MUSE
    muse = MUSE(random_state=0, window_inc=4, use_first_order_differences=False)
    muse.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = muse.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, muse_basic_motions_probas, decimal=2)


muse_unit_test_probas = np.array(
    [
        [
            0.5188702628480195,
            0.48112973715198043,
        ],
        [
            0.6356096946273173,
            0.36439030537268274,
        ],
        [
            0.07671090583536311,
            0.9232890941646369,
        ],
        [
            0.9777183534209937,
            0.022281646579006317,
        ],
        [
            0.8318769068654864,
            0.16812309313451354,
        ],
        [
            0.8783136574895891,
            0.12168634251041094,
        ],
        [
            0.7741193105790317,
            0.22588068942096828,
        ],
        [
            0.05536101985998121,
            0.9446389801400188,
        ],
        [
            0.939336763271401,
            0.06066323672859898,
        ],
        [
            0.8893828431193025,
            0.11061715688069752,
        ],
    ]
)
muse_basic_motions_probas = np.array(
    [
        [
            0.001927246919166307,
            0.0012708369244550188,
            0.0008872622012755287,
            0.9959146539551031,
        ],
        [
            0.8211527951523602,
            0.07957113879317383,
            0.03358900612067552,
            0.0656870599337905,
        ],
        [
            0.011051238492164021,
            0.009804574986833877,
            0.9601770167483709,
            0.018967169772631153,
        ],
        [
            0.05756962007782241,
            0.9091263179678627,
            0.014315070469148804,
            0.018988991485165994,
        ],
        [
            0.0032305688635647185,
            0.004852527299624302,
            0.002070751187255294,
            0.9898461526495558,
        ],
        [
            0.003575683952312505,
            0.0016386627477229351,
            0.0025945444799170426,
            0.9921911088200475,
        ],
        [
            0.8459487368337597,
            0.06981681985452316,
            0.03278588344190493,
            0.05144855986981213,
        ],
        [
            0.056241127789042104,
            0.018647918146872427,
            0.8846218478685607,
            0.04048910619552475,
        ],
        [
            0.05730640978991306,
            0.9248932752127209,
            0.010206816407129811,
            0.007593498590236224,
        ],
        [
            0.010569675570434757,
            0.983280248033324,
            0.003985513262902589,
            0.0021645631333385563,
        ],
    ]
)

# def print_array(array):
#     print('[')
#     for sub_array in array:
#         print('[')
#         for value in sub_array:
#             print(value.astype(str), end='')
#             print(', ')
#         print('],')
#     print(']')
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     muse_u = MUSE(random_state=0, window_inc=4, use_first_order_differences=False)
#
#     muse_u.fit(X_train, y_train)
#     probas = muse_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     muse_m = MUSE(random_state=0, window_inc=4, use_first_order_differences=False)
#
#     muse_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = muse_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
