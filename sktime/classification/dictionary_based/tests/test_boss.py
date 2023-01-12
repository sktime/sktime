# -*- coding: utf-8 -*-
"""BOSS test code."""
import numpy as np
import pytest

from sktime.classification.dictionary_based import BOSSEnsemble, IndividualBOSS
from sktime.datasets import load_unit_test


@pytest.fixture
def dataset():
    """
    Load unit_test train and test data set from sktime.

    :return: tuple, (X_train, y_train, X_test, y_test).
    """
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    return (X_train, y_train, X_test, y_test)


def test_individual_boss_classes_string(dataset):
    """Test of Individual Boss on unit test data with class dtype as STRING."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": "Class1", "2": "Class2"}
    y_train = np.array([new_class[y] for y in y_train])

    # train iboss and predict X_test
    iboss = IndividualBOSS()
    iboss.fit(X_train, y_train)
    y_pred = iboss.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == object
    assert set(y_pred) == set(y_train)


def test_individual_boss_classes_integer(dataset):
    """Test of Individual Boss on unit test data with class dtype as INTEGER."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": 1, "2": 2}
    y_train = np.array([new_class[y] for y in y_train])

    # train iboss and predict X_test
    iboss = IndividualBOSS()
    iboss.fit(X_train, y_train)
    y_pred = iboss.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == int
    assert set(y_pred) == set(y_train)


def test_individual_boss_classes_float(dataset):
    """Test of Individual Boss on unit test data with class dtype as FLOAT."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": 1.0, "2": 2.0}
    y_train = np.array([new_class[y] for y in y_train])

    # train iboss and predict X_test
    iboss = IndividualBOSS()
    iboss.fit(X_train, y_train)
    y_pred = iboss.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == float
    assert set(y_pred) == set(y_train)


def test_individual_boss_classes_boolean(dataset):
    """Test of Individual Boss on unit test data with class dtype as BOOLEAN."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": True, "2": False}
    y_train = np.array([new_class[y] for y in y_train])

    # train iboss and predict X_test
    iboss = IndividualBOSS()
    iboss.fit(X_train, y_train)
    y_pred = iboss.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == bool
    assert set(y_pred) == set(y_train)


def test_boss_ensemble_classes_string(dataset):
    """Test of Boss Ensemble on unit test data with class dtype as STRING."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": "Class1", "2": "Class2"}
    y_train = np.array([new_class[y] for y in y_train])

    # train boss_ensemble and predict X_test
    boss_ensemble = BOSSEnsemble()
    boss_ensemble.fit(X_train, y_train)
    y_pred = boss_ensemble.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == "<U6"
    assert set(y_pred) == set(y_train)


def test_boss_ensemble_classes_integer(dataset):
    """Test of Boss Ensemble on unit test data with class dtype as INTEGER."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": 1, "2": 2}
    y_train = np.array([new_class[y] for y in y_train])

    # train boss_ensemble and predict X_test
    boss_ensemble = BOSSEnsemble()
    boss_ensemble.fit(X_train, y_train)
    y_pred = boss_ensemble.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == int
    assert set(y_pred) == set(y_train)


def test_boss_ensemble_classes_float(dataset):
    """Test of Boss Ensemble on unit test data with class dtype as FLOAT."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": 1.0, "2": 2.0}
    y_train = np.array([new_class[y] for y in y_train])

    # train boss_ensemble and predict X_test
    boss_ensemble = BOSSEnsemble()
    boss_ensemble.fit(X_train, y_train)
    y_pred = boss_ensemble.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == float
    assert set(y_pred) == set(y_train)


def test_boss_ensemble_classes_boolean(dataset):
    """Test of Boss Ensemble on unit test data with class dtype as BOOLEAN."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    new_class = {"1": True, "2": False}
    y_train = np.array([new_class[y] for y in y_train])

    # train boss_ensemble and predict X_test
    boss_ensemble = BOSSEnsemble()
    boss_ensemble.fit(X_train, y_train)
    y_pred = boss_ensemble.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == bool
    assert set(y_pred) == set(y_train)
