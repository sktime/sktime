import numpy as np
from numpy import testing

from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.datasets import load_gunpoint, load_basic_motions


def test_cif_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train cif
    cif = CanonicalIntervalForest(n_estimators=100, random_state=0)
    cif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_gunpoint_probas)


def test_cif_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split='train', return_X_y=True)
    X_test, y_test = load_basic_motions(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train cif
    cif = CanonicalIntervalForest(n_estimators=100, random_state=0)
    cif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_basic_motions_probas)


cif_gunpoint_probas = np.array([
    [0.05, 0.95, ],
    [0.2, 0.8, ],
    [0.26, 0.74, ],
    [0.13, 0.87, ],
    [0.07, 0.93, ],
    [0.94, 0.06, ],
    [0.04, 0.96, ],
    [0.58, 0.42, ],
    [0.57, 0.43, ],
    [0.04, 0.96, ],
])
cif_basic_motions_probas = np.array([
    [1.0, 0.0, ],
    [0.26, 0.74, ],
    [0.49, 0.51, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.26, 0.74, ],
    [1.0, 0.0, ],
    [0.14, 0.86, ],
    [0.0, 1.0, ],
    [0.0, 1.0, ],
    [0.87, 0.13, ],
    [0.0, 1.0, ],
    [0.0, 1.0, ],
    [0.56, 0.44, ],
    [0.99, 0.01, ],
    [0.19, 0.81, ],
    [0.34, 0.66, ],
    [0.49, 0.51, ],
    [1.0, 0.0, ],
])


def print_array(array):
    print('[')
    for sub_array in array:
        print('[', end='')
        for value in sub_array:
            print(value.astype(str), end='')
            print(', ', end='')
        print('],')
    print(']')


if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    cif_u = CanonicalIntervalForest(n_estimators=100, random_state=0)

    cif_u.fit(X_train.iloc[indices], y_train[indices])
    probas = cif_u.predict_proba(X_test.iloc[indices])
    print_array(probas)

    X_train, y_train = load_basic_motions(split='train', return_X_y=True)
    X_test, y_test = load_basic_motions(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    cif_m = CanonicalIntervalForest(n_estimators=100, random_state=0)

    cif_m.fit(X_train.iloc[indices], y_train[indices])
    probas = cif_m.predict_proba(X_test.iloc[indices])
    print_array(probas)
