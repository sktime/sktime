# -*- coding: utf-8 -*-
"""ShapeletTransformClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_japanese_vowels, load_unit_test


def test_stc_on_unit_test_data():
    """Test of ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train STC
    rotf = RotationForest(n_estimators=10)
    stc = ShapeletTransformClassifier(
        # n_shapelets=100,
        transform_limit_in_minutes=0.025,
        random_state=0,
        estimator=rotf,
    )
    stc.fit(X_train, y_train)

    # assert probabilities are the same
    probas = stc.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, stc_unit_test_probas)

    # test train estimate
    train_probas = stc._get_train_probs(X_train, y_train)
    train_preds = stc.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


def test_contracted_stc_on_unit_test_data():
    """Test of contracted ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)

    # train contracted STC todo fix and comment out
    stc = ShapeletTransformClassifier(
        # time_limit_in_minutes=0.025,
        transform_limit_in_minutes=0.025,
        random_state=0,
    )
    stc.fit(X_train, y_train)

    # assert len(stc.estimators_) > 1
    assert accuracy_score(y_test, stc.predict(X_test)) >= 0.75


def test_stc_on_japanese_vowels():
    """Test of ShapeletTransformClassifier on japanese vowels data."""
    # load japanese vowels data
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)
    indices = [0, 1, 2, 3, 31, 32, 33, 34, 69, 70, 71, 72]

    # train STC
    rotf = RotationForest(n_estimators=10)
    stc = ShapeletTransformClassifier(
        n_shapelets=100,
        random_state=0,
        estimator=rotf,
    )
    stc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = stc.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, stc_japanese_vowels_probas)


stc_unit_test_probas = np.array(
    [
        [
            0.3,
            0.7,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7,
            0.3,
        ],
    ]
)
stc_japanese_vowels_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.5,
            0.5,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.6,
            0.4,
        ],
        [
            0.2,
            0.7,
            0.0,
            0.1,
        ],
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.0,
            0.1,
            0.3,
            0.6,
        ],
        [
            0.7,
            0.3,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.1,
            0.6,
            0.3,
        ],
        [
            0.3,
            0.6,
            0.0,
            0.1,
        ],
        [
            0.3,
            0.7,
            0.0,
            0.0,
        ],
    ]
)


# def print_array(array):
#     print("[")
#     for sub_array in array:
#         print("[")
#         for value in sub_array:
#             print(value.astype(str), end="")
#             print(", ")
#         print("],")
#     print("]")
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     rf = RandomForestClassifier(n_estimators=10)
#     stc_u = ShapeletTransformClassifier(
#         # n_shapelets=100,
#         transform_limit_in_minutes=0.025,
#         random_state=0,
#         estimator=rf,
#     )
#
#     stc_u.fit(X_train, y_train)
#     probas = stc_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
#     X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     rf = RandomForestClassifier(n_estimators=10)
#     stc_m = ShapeletTransformClassifier(
#         n_shapelets=100,
#         random_state=0,
#         estimator=rf,
#     )
#
#     stc_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = stc_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
