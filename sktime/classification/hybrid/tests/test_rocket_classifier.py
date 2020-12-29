# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.hybrid import ROCKETClassifier
from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions


def test_rocket_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train TDE
    rocket = ROCKETClassifier(random_state=0)
    rocket.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, rocket_gunpoint_probas)


def test_rocket_ensemble_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train IndividualTDE
    rocket_e = ROCKETClassifier(
        num_kernels=2000,
        ensemble=True,
        random_state=0,
    )
    rocket_e.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket_E.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, rocket_e_gunpoint_probas)


def test_tde_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train TDE
    rocket = ROCKETClassifier(random_state=0)
    rocket.fit(X_train, y_train)

    score = rocket.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_tde_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train c22f
    rocket = ROCKETClassifier(random_state=0)
    rocket.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, rocket_basic_motions_probas)
